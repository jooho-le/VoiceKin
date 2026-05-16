import "./styles.css";

function getDefaultApiBase() {
  const { protocol, hostname } = window.location;

  if (protocol === "http:" && (hostname === "127.0.0.1" || hostname === "localhost")) {
    return "http://127.0.0.1:8000";
  }
  if (protocol === "http:" && hostname) {
    return `http://${hostname}:8000`;
  }
  return "http://10.0.2.2:8000";
}

function getInitialApiBase() {
  const savedApiBase = localStorage.getItem("voicekin_api_base");
  const { protocol, hostname } = window.location;
  const isLocalWebPreview =
    protocol === "http:" && (hostname === "127.0.0.1" || hostname === "localhost");
  const isAndroidAppPreview = protocol === "https:" && hostname === "localhost";

  if (!savedApiBase) {
    return getDefaultApiBase();
  }

  // 웹 미리보기에서 Android 에뮬레이터용 주소가 저장되어 있으면 서버 연결이 실패한다.
  // 반대로 Android 앱에서는 PC 브라우저용 127.0.0.1 주소가 자기 자신을 가리키므로 실패한다.
  if (isLocalWebPreview && savedApiBase.includes("10.0.2.2")) {
    return "http://127.0.0.1:8000";
  }
  if (isAndroidAppPreview && savedApiBase.includes("127.0.0.1")) {
    return "http://10.0.2.2:8000";
  }

  return savedApiBase;
}

const defaultApiBase = getDefaultApiBase();
const state = {
  apiBase: getInitialApiBase(),
  callFile: null,
  callerName: localStorage.getItem("voicekin_caller_name") || "엄마",
  startedAt: null,
  timer: null,
  warningTimer: null,
  analysisPromise: null,
  latestAnalysis: null,
  setupCollapsed: false,
};

const $ = (id) => document.getElementById(id);

function apiUrl(path) {
  return `${state.apiBase.replace(/\/$/, "")}${path}`;
}

function setText(id, value) {
  $(id).textContent = value;
}

function setCallState(message) {
  setText("callState", message);
}

function setLiveMessage(message) {
  setText("liveMessage", message);
}

function setSetupHint(message) {
  setText("setupHint", message);
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch (error) {
    throw new Error(`AI 서버 연결 실패: ${state.apiBase}`);
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return payload;
}

async function checkServer() {
  $("apiBaseInput").value = state.apiBase;
  $("callerNameInput").value = state.callerName;
  applyCallerName();

  try {
    await requestJson("/health");
    setText("networkStatus", "VoiceKin 연결됨");
  } catch {
    setText("networkStatus", "서버 연결 필요");
  }
}

function saveSetup() {
  state.apiBase = $("apiBaseInput").value.trim() || defaultApiBase;
  state.callerName = $("callerNameInput").value.trim() || "엄마";
  localStorage.setItem("voicekin_api_base", state.apiBase);
  localStorage.setItem("voicekin_caller_name", state.callerName);
  applyCallerName();
  checkServer();
  setSetupHint("설정이 저장됐습니다.");
}

function applyCallerName() {
  setText("callerName", state.callerName);
  setText("avatar", state.callerName.trim().charAt(0) || "가");
}

function updateFileLabel() {
  state.callFile = $("callAudioFile").files[0] || null;
  setText("callFileName", state.callFile ? state.callFile.name : "파일 선택");
}

function prepareCall() {
  saveSetup();
  updateFileLabel();

  if (!state.callFile) {
    setSetupHint("통화 음성 파일을 먼저 선택하세요.");
    return;
  }

  $("warningPopup").classList.add("hidden");
  state.latestAnalysis = null;
  state.analysisPromise = null;
  setCallState("통화 준비 완료");
  setLiveMessage("통화 시작 버튼을 누르면 선택한 음성이 재생됩니다.");
  setSetupHint("준비됐습니다. 통화 화면의 재생 버튼을 누르세요.");
  collapseSetup(true);
}

function startCall() {
  if (!state.callFile) {
    prepareCall();
  }
  if (!state.callFile) return;

  $("warningPopup").classList.add("hidden");
  setCallState("통화 중");
  setLiveMessage("VoiceKin이 보이스피싱인지 분석중 입니다.");

  const audio = $("callAudioPlayer");
  audio.src = URL.createObjectURL(state.callFile);
  audio.play().catch(() => {
    setLiveMessage("재생 버튼을 한 번 더 누르면 음성이 재생됩니다.");
  });

  state.startedAt = Date.now();
  if (state.timer) clearInterval(state.timer);
  if (state.warningTimer) clearTimeout(state.warningTimer);
  state.timer = setInterval(updateTimer, 300);

  state.analysisPromise = analyzeFile(state.callFile)
    .then((data) => {
      state.latestAnalysis = data;
      return data;
    })
    .catch((error) => {
      state.latestAnalysis = { error: error.message };
      return state.latestAnalysis;
    });

  state.warningTimer = setTimeout(showAnalysisPopup, 9000);
}

function stopCall() {
  const audio = $("callAudioPlayer");
  audio.pause();
  audio.removeAttribute("src");

  if (state.timer) clearInterval(state.timer);
  if (state.warningTimer) clearTimeout(state.warningTimer);

  state.startedAt = null;
  setText("callTimer", "00:00");
  setCallState("통화 종료");
  setLiveMessage("통화를 종료했습니다. 다시 시작하려면 통화 준비를 누르세요.");
}

function updateTimer() {
  if (!state.startedAt) return;
  const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  setText("callTimer", `${mm}:${ss}`);
}

async function showAnalysisPopup() {
  setCallState("VoiceKin 확인 중");

  const data = state.latestAnalysis || await state.analysisPromise;
  if (!data || data.error) {
    showWarning({
      title: "분석 지연",
      message: data?.error || "서버 분석이 아직 완료되지 않았습니다.",
      safe: false,
    });
    return;
  }

  const warning = buildWarning(data);
  showWarning(warning);
}

function buildWarning(data) {
  const family = data.family_verification;
  const anti = data.anti_spoofing;
  const bestMatch = family.best_match;
  const displayName = bestMatch?.name || state.callerName || "등록 가족";

  if (anti.is_spoofed && !family.is_registered_family) {
    return {
      title: "보이스피싱이 의심됩니다",
      message: `저장된 ${displayName}의 목소리와 다르고 AI 합성 음성 신호가 감지됐습니다. 통화를 중단하고 직접 확인하세요.`,
      safe: false,
    };
  }

  if (!family.is_registered_family) {
    return {
      title: "보이스피싱이 의심됩니다",
      message: `저장된 ${displayName}의 목소리와 일치하지 않습니다. 통화 내용을 믿기 전에 가족에게 다시 확인하세요.`,
      safe: false,
    };
  }

  if (anti.is_spoofed) {
    return {
      title: "AI 음성 의심",
      message: `${displayName}와 비슷하지만 AI 합성 음성 신호가 감지됐습니다. 민감한 정보 요청은 거절하세요.`,
      safe: false,
    };
  }

  return {
    title: "위험 신호가 낮습니다",
    message: `저장된 ${displayName}의 목소리와 일치합니다. 그래도 송금이나 인증번호 요청은 한 번 더 확인하세요.`,
    safe: true,
  };
}

function showWarning({ title, message, safe }) {
  $("warningPopup").classList.toggle("safe", safe);
  $("warningPopup").classList.remove("hidden");
  setText("warningTitle", title);
  setText("warningMessage", message);
  setCallState(safe ? "VoiceKin 정상 확인" : "VoiceKin 경고");
  setLiveMessage(message);
}

async function analyzeFile(file) {
  const formData = new FormData();
  formData.append("audio_file", file);
  return requestJson("/api/v1/voice/verify-family-secure", {
    method: "POST",
    body: formData,
  });
}

function collapseSetup(forceValue = null) {
  state.setupCollapsed = forceValue === null ? !state.setupCollapsed : forceValue;
  $("setupSheet").classList.toggle("collapsed", state.setupCollapsed);
  $("setupBody").classList.toggle("hidden", state.setupCollapsed);
  $("toggleSetupBtn").textContent = state.setupCollapsed ? "설정" : "접기";
}

function bindEvents() {
  $("saveSetupBtn").addEventListener("click", saveSetup);
  $("prepareCallBtn").addEventListener("click", prepareCall);
  $("startCallBtn").addEventListener("click", startCall);
  $("stopCallBtn").addEventListener("click", stopCall);
  $("toggleSetupBtn").addEventListener("click", () => collapseSetup());
  $("callAudioFile").addEventListener("change", updateFileLabel);
}

bindEvents();
checkServer();
