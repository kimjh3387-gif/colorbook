// 색칠공부 변환기 — 모든 처리는 브라우저 안에서만 일어난다 (업로드/네트워크 요청 없음)
(() => {
  const MAX_DIMENSION = 1600; // 성능 보호용 최대 변 길이
  // 작은 원본(이미지 검색 썸네일을 우클릭 저장한 400~600px짜리가 대부분)은 처리 전에 이 크기로 확대한다.
  // 안 그러면 단순화 3(블러 반경)이 작은 그림에선 귀·얼굴선 같은 요소를 통째로 지워버리고,
  // 결과 PNG도 인쇄하기엔 너무 작다. 확대해 두면 슬라이더 감각이 큰 사진과 같아진다.
  const MIN_DIMENSION = 1200;

  // ---------- AI 윤곽 모드 (PiDiNet) ----------
  // 그래디언트 방식은 "이미 보이는 경계"만 뽑는다. 3D 렌더링 캐릭터의 팔처럼 색은 같고 그림자만
  // 16단계쯤 다른 경계는 신호 자체가 없어서(피카츄 실측) 임계값을 아무리 내려도 못 살린다.
  // PiDiNet(ICCV 2021)은 사람이 그린 윤곽(BSDS500)으로 학습된 경량 엣지 모델이라 그런 경계를 "안다".
  // tiny 버전(0.4MB)을 onnxruntime-web으로 브라우저 안에서 돌린다 — 서버·API 키 없음, 이미지 안 나감.
  // 실측(1024px): WebGPU 첫 실행 2.7초(셰이더 컴파일), 이후 0.25초. WASM 폴백은 ~1초.
  const AI_MODEL_URL = 'models/pidinet_tiny.onnx';
  const ORT_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js';
  const AI_OUTPUT_DIMENSION = 1024; // 결과 PNG 긴 변 (추론 해상도와 무관하게 고정)
  function getBackgroundSession() {
    if (bgSessionPromise) return bgSessionPromise;
    bgSessionPromise = (async () => {
      await loadScript(ORT_SCRIPT_URL);
      return ort.InferenceSession.create(BG_MODEL_URL, { executionProviders: ['webgpu', 'wasm'] });
    })();
    bgSessionPromise.catch(() => { bgSessionPromise = null; });
    return bgSessionPromise;
  }

  // 원본 → 배경을 흰색으로 밀어낸 캔버스(원본 크기). 원본당 한 번만 계산.
  async function getBackgroundRemoved() {
    if (bgCache && bgCache.image === sourceImage) return bgCache.canvas;
    const image = sourceImage;

    if (!bgSessionPromise) processingOverlay.textContent = '배경 분리 모델 준비 중… (처음 한 번만)';
    const session = await getBackgroundSession();
    processingOverlay.textContent = '배경 지우는 중…';

    // 320×320 입력 (비율 무시하고 늘림 — 모델 학습 방식과 같고, 마스크를 다시 원본 비율로 늘리면 맞는다)
    const c = document.createElement('canvas');
    c.width = BG_INPUT;
    c.height = BG_INPUT;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(image, 0, 0, BG_INPUT, BG_INPUT);
    const d = ctx.getImageData(0, 0, BG_INPUT, BG_INPUT).data;
    const n = BG_INPUT * BG_INPUT;
    const x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let ch = 0; ch < 3; ch++) {
        x[ch * n + i] = (d[i * 4 + ch] / 255 - AI_MEAN[ch]) / AI_STD[ch];
      }
    }
    const out = await session.run({ 'input.1': new ort.Tensor('float32', x, [1, 3, BG_INPUT, BG_INPUT]) });
    const raw = out[session.outputNames[0]].data; // 첫 출력(d0)이 최종 마스크
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { if (raw[i] < lo) lo = raw[i]; if (raw[i] > hi) hi = raw[i]; }
    const span = hi - lo || 1;

    // 마스크를 회색 이미지로 만들어 원본 크기로 부드럽게 키운 뒤 알파로 쓴다.
    // 0.15~0.85 구간을 0~1로 펴서(그 밖은 확실히 배경/피사체) 경계가 뿌옇게 남지 않게 한다.
    const mc = document.createElement('canvas');
    mc.width = BG_INPUT;
    mc.height = BG_INPUT;
    const mctx = mc.getContext('2d');
    const mimg = mctx.createImageData(BG_INPUT, BG_INPUT);
    for (let i = 0; i < n; i++) {
      let m = (raw[i] - lo) / span;
      m = clamp((m - 0.15) / 0.7, 0, 1);
      const v = Math.round(m * 255);
      mimg.data[i * 4] = v; mimg.data[i * 4 + 1] = v; mimg.data[i * 4 + 2] = v; mimg.data[i * 4 + 3] = 255;
    }
    mctx.putImageData(mimg, 0, 0);

    const W = image.naturalWidth;
    const H = image.naturalHeight;
    const big = document.createElement('canvas');
    big.width = W;
    big.height = H;
    const bctx = big.getContext('2d', { willReadFrequently: true });
    bctx.imageSmoothingQuality = 'high';
    bctx.drawImage(mc, 0, 0, W, H);
    const mask = bctx.getImageData(0, 0, W, H).data;

    const outCanvas = document.createElement('canvas');
    outCanvas.width = W;
    outCanvas.height = H;
    const octx = outCanvas.getContext('2d', { willReadFrequently: true });
    octx.drawImage(image, 0, 0);
    const img = octx.getImageData(0, 0, W, H);
    const p = img.data;
    for (let i = 0; i < p.length; i += 4) {
      const a = mask[i] / 255;
      p[i] = Math.round(p[i] * a + 255 * (1 - a));
      p[i + 1] = Math.round(p[i + 1] * a + 255 * (1 - a));
      p[i + 2] = Math.round(p[i + 2] * a + 255 * (1 - a));
    }
    octx.putImageData(img, 0, 0);

    bgCache = { image, canvas: outCanvas };
    return outCanvas;
  }

  // 추론 해상도 = AI 모드의 "단순화". 모델에 작은 그림을 주면 작은 주름은 안 보이고 큰 윤곽만 잡는다.
  // 실측: 인형 사진은 1024에서 봉제선·털 주름을 전부 선으로 잡았고 512~768에서 얼굴은 그대로 두고
  // 몸통 잔선만 빠졌다. 렌더 피카츄는 768이 1024보다 팔 위쪽이 더 잘 이어졌다(1024는 점선).
  // 단순화 0.5→1024, 3(기본)→768, 6.5→512, 12.5→256: 6단계마다 절반.
  const AI_MAX_DIMENSION = 1024;
  const AI_MIN_DIMENSION = 256;
  const AI_MEAN = [0.485, 0.456, 0.406]; // ImageNet 정규화 (모델 학습 조건)
  const AI_STD = [0.229, 0.224, 0.225];

  // ---------- 배경 제거 (U²-Net-p) ----------
  // 숲 배경 애니 장면처럼 배경이 있으면 어떤 모드든 배경 텍스처가 선으로 딸려온다. 그래서 변환 전에
  // 주요 피사체만 남기고 배경을 흰색으로 밀어낸다. U²-Net-p(4.6MB, Apache-2.0 → 광고 붙여도 됨)를
  // 320×320으로 돌려 마스크를 얻고 원본 크기로 키워 합성. 실측 CPU 150~190ms.
  // 한계: 경량 모델이라 "주인공 하나" 위주 — 옆에 있는 두 번째 캐릭터는 같이 지워질 수 있다.
  const BG_MODEL_URL = 'models/u2netp.onnx';
  const BG_INPUT = 320;

  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const controls = document.getElementById('controls');
  const previewEmpty = document.getElementById('previewEmpty');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const uploadThumb = document.getElementById('uploadThumb');
  const previewGrid = document.getElementById('previewGrid');
  const originalCanvas = document.getElementById('originalCanvas');
  const resultCanvas = document.getElementById('resultCanvas');
  const processingOverlay = document.getElementById('processingOverlay');
  const downloadBtn = document.getElementById('downloadBtn');
  const resetBtn = document.getElementById('resetBtn');
  const newImageBtn = document.getElementById('newImageBtn');

  const sensitivity = document.getElementById('sensitivity');
  const thickness = document.getElementById('thickness');
  const denoise = document.getElementById('denoise');
  const adaptive = document.getElementById('adaptive');
  const shading = document.getElementById('shading');
  const prune = document.getElementById('prune');
  const pruneVal = document.getElementById('pruneVal');
  const modeInputs = document.querySelectorAll('input[name="mode"]');
  const autoHint = document.getElementById('autoHint');
  const outlineAssist = document.getElementById('outlineAssist');
  const inkDark = document.getElementById('inkDark');
  const inkDarkVal = document.getElementById('inkDarkVal');
  const titleInput = document.getElementById('titleInput');
  const printBtn = document.getElementById('printBtn');
  const printResult = document.getElementById('printResult');
  const printOriginal = document.getElementById('printOriginal');
  const printTitle = document.getElementById('printTitle');
  const sensitivityVal = document.getElementById('sensitivityVal');
  const thicknessVal = document.getElementById('thicknessVal');
  const denoiseVal = document.getElementById('denoiseVal');
  const adaptiveVal = document.getElementById('adaptiveVal');

  const eraserBtn = document.getElementById('eraserBtn');
  const eraserSizeWrap = document.getElementById('eraserSizeWrap');
  const eraserSize = document.getElementById('eraserSize');
  const undoBtn = document.getElementById('undoBtn');
  const clearEraseBtn = document.getElementById('clearEraseBtn');
  const resultWrap = document.getElementById('resultWrap');
  const eraserCursor = document.getElementById('eraserCursor');
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxClose = document.getElementById('lightboxClose');

  // denoise 기본값 3: 6이면 해·광배처럼 가늘고 대비 약한 요소가 흐림 단계에서 통째로 지워진다
  // (측정: 단순화 2~4에서는 해가 잡히고 6에서는 0). 3이면 텍스처는 눌리면서 해는 살아남는다.
  // sensitivity 기본값 70: 컬러 그래디언트로 바꾼 뒤 흰 배경 실루엣이 워낙 세게 잡혀서 상위 n%를
  // 독식한다. 50이면 실루엣만 남고 눈·볼·입이 빠지고, 65면 입이 빠지고, 70에서 얼굴이 다 들어온다
  // (피카츄 썸네일 실측). 75부터는 JPEG 링잉이 손·발 근처에 꼬불선으로 나타나기 시작한다.
  const DEFAULTS = { sensitivity: 70, thickness: 1, denoise: 3, adaptive: 60, shading: false, prune: 40, inkDark: 150 };

  let sourceImage = null; // HTMLImageElement
  let rendering = false;      // 렌더가 진행 중 (AI 모드는 GPU를 기다리는 동안 입력 이벤트가 계속 들어온다)
  let rerunRequested = false; // 진행 중에 슬라이더가 움직였으면 끝나고 한 번 더
  let aiSessionPromise = null; // onnxruntime 세션 (한 번만 만든다)
  let aiCache = null;          // { image, width, height, fused } — 슬라이더만 바꿀 땐 모델을 다시 안 돌린다
  let bgSessionPromise = null; // 배경 제거 세션
  let bgCache = null;          // { image, canvas } — 원본당 한 번만 배경을 지운다
  let activeSource = null;     // 이번 렌더가 실제로 읽는 소스 (원본 이미지 또는 배경 지운 캔버스)
  let assistDecidedFor = null; // 윤곽선 보충 자동 판단을 이미 내린 소스 (한 그림에 한 번만)
  const bgRemove = document.getElementById('bgRemove');

  // 소스가 <img>든 <canvas>든 같은 방식으로 크기를 읽는다
  function sourceSize(src) {
    return src instanceof HTMLCanvasElement
      ? { w: src.width, h: src.height }
      : { w: src.naturalWidth, h: src.naturalHeight };
  }

  // ---------- 지우개 상태 ----------
  // 알고리즘이 못 가리는 선(인형 봉제선 등)은 사람이 지운다. 지운 자국은 캔버스 크기에 대한 비율 좌표로
  // 저장해서, 슬라이더를 다시 만져 결과가 새로 그려지거나(=paintMask) 모드가 바뀌어 캔버스 크기가
  // 달라져도 같은 자리에 다시 적용된다. 새 이미지를 올리면 비운다.
  let eraseStrokes = [];   // [{ r: 반지름(폭 대비 비율), pts: [[x,y], ...] (0~1) }]
  let lastPaint = null;    // 마지막으로 그린 { mask, shadeMask, width, height } — 되돌리기용 재도색
  let eraserOn = false;
  let activeStroke = null;

  function currentMode() {
    return document.querySelector('input[name="mode"]:checked').value;
  }

  function setMode(mode) {
    modeInputs.forEach((input) => { input.checked = input.value === mode; });
    controls.dataset.mode = mode;
  }

  // ---------- 그림 종류 자동 감지 ----------
  // 검은 윤곽선이 이미 그려진 만화·일러스트는 엣지 검출(선이 두 줄로 갈라짐)도 AI(작은 선이 뭉개짐)도
  // 손해고, 검은 픽셀을 그대로 뽑는 '선화' 모드가 정답이다. 판별은 세 가지 (400px로 줄여서 측정):
  //   1) 어두운 픽셀(RGB 최대값 < 70) 비율 ≥ 1%
  //   2) 흰 배경(RGB 최소값 > 235) 비율 ≥ 25%  — 숲 배경 애니 장면(3.5%)을 걸러냄
  //   3) 어두운 픽셀이 "면"이 아니라 "선"일 것: 반지름 (긴 변/150)로 침식하면 75% 넘게 사라져야 함
  //      — 귀 끝이 검게 칠해진 인형/렌더(61%·56% 남음)를 걸러냄
  // 실측: 일러스트 1.55%/52.7%/8.6% ○, 카드 격자 1.13%/42%/13% ○, 인형 1.56%/65%/61% ✕,
  //       렌더 0.63%/71%/56% ✕, 애니 장면 8.9%/3.5%/27% ✕
  function looksLikeInkDrawing(image) {
    const longest = 400;
    const { w: iw, h: ih } = sourceSize(image);
    const scale = Math.min(1, longest / Math.max(iw, ih));
    const w = Math.max(8, Math.round(iw * scale));
    const h = Math.max(8, Math.round(ih * scale));
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data;
    const dark = new Uint8Array(w * h);
    let darkCount = 0;
    let whiteCount = 0;
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      if (Math.max(d[i], d[i + 1], d[i + 2]) < 70) { dark[p] = 1; darkCount++; }
      if (Math.min(d[i], d[i + 1], d[i + 2]) > 235) whiteCount++;
    }
    if (darkCount / (w * h) < 0.01) return false;
    if (whiteCount / (w * h) < 0.25) return false;
    // 침식 = 반전 → 팽창 → 반전
    const inv = new Uint8Array(w * h);
    for (let i = 0; i < inv.length; i++) inv[i] = dark[i] ? 0 : 1;
    const grown = dilateDisc(inv, w, h, Math.max(2, Math.round(Math.max(w, h) / 150)));
    let remain = 0;
    for (let i = 0; i < grown.length; i++) if (dark[i] && !grown[i]) remain++;
    return remain / darkCount <= 0.25;
  }

  // ---------- 업로드 ----------
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFile(file);
  });
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) loadFile(file);
  });
  newImageBtn.addEventListener('click', () => fileInput.click());

  function loadFile(file) {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      sourceImage = img;
      URL.revokeObjectURL(url);
      eraseStrokes = [];
      updateEraserButtons();
      drawOriginal();
      assistDecidedFor = null; // 새 그림 → 윤곽선 보충 필요 여부를 다시 판단 (배경 지운 뒤에)
      uploadPrompt.hidden = true;
      uploadThumb.hidden = false;
      controls.hidden = false;
      previewEmpty.hidden = true;
      previewGrid.hidden = false;
      scheduleRender();
    };
    img.src = url;
  }

  function drawOriginal() {
    const { width, height } = fitSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
    originalCanvas.width = width;
    originalCanvas.height = height;
    const ctx = originalCanvas.getContext('2d');
    ctx.drawImage(sourceImage, 0, 0, width, height);
  }

  function fitSize(w, h) {
    const longest = Math.max(w, h);
    let scale = 1;
    if (longest > MAX_DIMENSION) scale = MAX_DIMENSION / longest;
    else if (longest < MIN_DIMENSION) scale = MIN_DIMENSION / longest;
    return { width: Math.round(w * scale), height: Math.round(h * scale) };
  }

  // ---------- 컨트롤 ----------
  [
    [sensitivity, sensitivityVal],
    [thickness, thicknessVal],
    [denoise, denoiseVal],
    [adaptive, adaptiveVal],
    [prune, pruneVal],
    [inkDark, inkDarkVal],
  ].forEach(([input, out]) => {
    input.addEventListener('input', () => {
      out.textContent = input.value;
      scheduleRender();
    });
  });

  shading.addEventListener('change', scheduleRender);
  bgRemove.addEventListener('change', scheduleRender);
  outlineAssist.addEventListener('change', () => { autoHint.hidden = true; scheduleRender(); });

  modeInputs.forEach((input) => {
    input.addEventListener('change', () => {
      controls.dataset.mode = currentMode();
      autoHint.hidden = true;
      scheduleRender();
    });
  });

  resetBtn.addEventListener('click', () => {
    sensitivity.value = DEFAULTS.sensitivity;
    thickness.value = DEFAULTS.thickness;
    denoise.value = DEFAULTS.denoise;
    adaptive.value = DEFAULTS.adaptive;
    prune.value = DEFAULTS.prune;
    pruneVal.textContent = DEFAULTS.prune;
    inkDark.value = DEFAULTS.inkDark;
    inkDarkVal.textContent = DEFAULTS.inkDark;
    shading.checked = DEFAULTS.shading;
    sensitivityVal.textContent = DEFAULTS.sensitivity;
    thicknessVal.textContent = DEFAULTS.thickness;
    denoiseVal.textContent = DEFAULTS.denoise;
    adaptiveVal.textContent = DEFAULTS.adaptive;
    setMode('ink');
    autoHint.hidden = true;
    scheduleRender();
  });

  // ---------- 크게 보기(돋보기) ----------
  document.querySelectorAll('.zoom-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation(); // 썸네일 쪽 돋보기는 업로드 칸 안에 있어서 파일 선택창이 같이 뜨지 않게
      const canvas = document.getElementById(btn.dataset.zoom);
      if (!canvas || !canvas.width) return;
      lightboxImg.src = canvas.toDataURL('image/png');
      lightbox.hidden = false;
    });
  });

  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImg.removeAttribute('src'); // 큰 data URL을 붙들고 있지 않도록
  }

  lightboxClose.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightbox(); // 배경 클릭
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !lightbox.hidden) closeLightbox();
  });

  downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'coloring-page.png';
    link.href = resultCanvas.toDataURL('image/png');
    link.click();
  });

  // ---------- 렌더 스케줄링 ----------
  // 렌더가 도는 동안 들어온 입력은 플래그 하나로 합쳐서, 끝난 뒤 마지막 상태로 한 번만 다시 돈다.
  // (예전엔 "큐에 있으면 무시"였는데, AI 모드는 GPU를 await 하는 동안 슬라이더 이벤트가 진짜로
  //  들어오기 때문에 마지막 위치가 버려지는 문제가 생긴다.)
  function scheduleRender() {
    if (!sourceImage) return;
    rerunRequested = true;
    if (rendering) return;
    rendering = true;
    processingOverlay.textContent = '변환 중…';
    processingOverlay.hidden = false;
    // requestAnimationFrame에 의존하면 탭이 백그라운드/비표시 상태일 때 콜백이 멈춘다.
    // setTimeout만 사용해 화면 표시 여부와 무관하게 실행되도록 한다.
    setTimeout(async () => {
      try {
        while (rerunRequested) {
          rerunRequested = false;
          activeSource = bgRemove.checked ? await getBackgroundRemoved() : sourceImage;
          if (rerunRequested) continue; // 배경 지우는 사이 입력이 바뀜 — 처음부터
          // 윤곽선 보충 자동 판단은 배경을 지운 뒤에 한다 (숲 배경이 사라지면 흰 비율이 올라가 판단이 정확해짐)
          if (assistDecidedFor !== activeSource) {
            assistDecidedFor = activeSource;
            const hasInk = looksLikeInkDrawing(activeSource);
            outlineAssist.checked = !hasInk;
            autoHint.textContent = hasInk
              ? '검은 윤곽선이 있는 그림이라 그 선을 그대로 써요'
              : '검은 선이 적은 그림이라 윤곽선을 보충했어요. 마음에 안 들면 꺼도 돼요';
            autoHint.hidden = false;
          }
          const mode = currentMode();
          if (mode === 'ink') await renderInk();
          else render();
        }
        processingOverlay.hidden = true;
      } catch (err) {
        // 에러가 나도 "변환 중"에 무한히 멈춰있지 않고 사용자에게 보여준다.
        console.error('색칠공부 변환 실패:', err);
        rerunRequested = false;
        processingOverlay.hidden = false;
        processingOverlay.textContent = '변환 중 오류가 났어요: ' + err.message;
      } finally {
        rendering = false;
      }
    }, 0);
  }

  // ---------- AI 윤곽 모드 ----------
  function loadScript(url) {
    return new Promise((resolve, reject) => {
      if (window.ort) { resolve(); return; }
      const el = document.createElement('script');
      el.src = url;
      el.onload = resolve;
      el.onerror = () => reject(new Error('AI 라이브러리를 못 불러왔어요 (인터넷 연결 확인)'));
      document.head.appendChild(el);
    });
  }

  function getAISession() {
    if (aiSessionPromise) return aiSessionPromise;
    aiSessionPromise = (async () => {
      await loadScript(ORT_SCRIPT_URL);
      // WebGPU가 되면 GPU, 아니면 WASM으로 자동 폴백 (순서대로 시도)
      return ort.InferenceSession.create(AI_MODEL_URL, { executionProviders: ['webgpu', 'wasm'] });
    })();
    aiSessionPromise.catch(() => { aiSessionPromise = null; }); // 실패하면 다음에 다시 시도할 수 있게
    return aiSessionPromise;
  }

  function aiInferenceDimension() {
    const d = parseFloat(denoise.value);
    const dim = AI_MAX_DIMENSION * Math.pow(0.5, (d - 0.5) / 6);
    return clamp(Math.round(dim / 8) * 8, AI_MIN_DIMENSION, AI_MAX_DIMENSION);
  }

  // 긴 변을 longest에 맞추고 각 변은 8의 배수로 (모델이 /8 다운샘플 후 다시 키우므로 안전하게)
  function fitSizeAI(w, h, longest) {
    const scale = longest / Math.max(w, h);
    const r = (v) => Math.max(8, Math.round(v * scale / 8) * 8);
    return { width: r(w), height: r(h) };
  }

  // 추론 크기의 확률 맵을 출력 크기로 부드럽게 키운다 (캔버스 bilinear). 8비트로 한 번 거치지만
  // 임계값 판정엔 충분하고, 작은 해상도로 돌린 결과도 결과 PNG는 항상 같은 크기가 된다.
  function upscaleMap(src, sw, sh, dw, dh) {
    if (sw === dw && sh === dh) return src;
    const c = document.createElement('canvas');
    c.width = sw;
    c.height = sh;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(sw, sh);
    for (let i = 0; i < src.length; i++) {
      const v = Math.round(src[i] * 255);
      img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const big = document.createElement('canvas');
    big.width = dw;
    big.height = dh;
    const bctx = big.getContext('2d', { willReadFrequently: true });
    bctx.imageSmoothingQuality = 'high';
    bctx.drawImage(c, 0, 0, dw, dh);
    const d = bctx.getImageData(0, 0, dw, dh).data;
    const out = new Float32Array(dw * dh);
    for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
    return out;
  }

  // 모델 출력(픽셀별 "여기가 윤곽일 확률" 0~1)을 (이미지, 추론 해상도)당 한 번만 계산해 캐시한다.
  async function getFusedMap() {
    const inferDim = aiInferenceDimension();
    if (aiCache && aiCache.image === activeSource && aiCache.inferDim === inferDim) return aiCache;
    const image = activeSource;

    if (!aiSessionPromise) processingOverlay.textContent = 'AI 모델 준비 중… (처음 한 번만)';
    const session = await getAISession();
    processingOverlay.textContent = 'AI가 윤곽을 찾는 중…';

    const { w: sw, h: sh } = sourceSize(image);
    const { width, height } = fitSizeAI(sw, sh, inferDim);
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(image, 0, 0, width, height);
    const d = ctx.getImageData(0, 0, width, height).data;

    const n = width * height;
    const x = new Float32Array(3 * n); // CHW
    for (let i = 0; i < n; i++) {
      for (let ch = 0; ch < 3; ch++) {
        x[ch * n + i] = (d[i * 4 + ch] / 255 - AI_MEAN[ch]) / AI_STD[ch];
      }
    }
    const out = await session.run({ image: new ort.Tensor('float32', x, [1, 3, height, width]) });
    const raw = Float32Array.from(out.fused.data, (v) => (v < 0 ? 0 : v > 1 ? 1 : v));

    const outSize = fitSizeAI(sw, sh, AI_OUTPUT_DIMENSION);
    const fused = upscaleMap(raw, width, height, outSize.width, outSize.height);

    aiCache = { image, inferDim, width: outSize.width, height: outSize.height, fused };
    return aiCache;
  }

  // 학습된 윤곽 모델로 "몸통 윤곽" 마스크를 만든다 (선화 모드의 윤곽선 보충). 결과는 1px 뼈대 → 굵기 확장.
  // 반환: { width, height, mask } 또는 null(기다리는 사이 소스가 바뀜)
  async function outlineMaskFromModel() {
    const { image, width, height, fused } = await getFusedMap();
    if (image !== activeSource) return null; // 기다리는 사이 다른 이미지로 바뀜 — 다음 루프가 처리

    // 출력이 확률이라 임계값이 사진마다 흔들리지 않는다. 0.2~0.5 사이는 거의 같은 그림
    // (실측 검은 비율 5.2%→4.1%). 70 기준으로 고정 — 슬라이더로 바꿔봐야 얻는 게 거의 없었다.
    const thr = 0.36;

    // 하드 임계값 하나면 확률이 오르내리는 약한 선(팔 위쪽 그림자 경계)이 점선으로 끊긴다.
    // 강한 선(thr 이상)에서 출발해 이어지는 약한 선(thr의 55%까지)은 살린다 — 기본 모드와 같은 원리.
    // 35%까지 내렸더니 인형 사진에서 털 주름이 줄줄이 딸려왔다. 추론 해상도를 768로 낮춘 뒤로는
    // 팔이 하드 임계값으로도 이어지므로 하한을 넉넉히 둘 필요가 없다.
    let mask = hysteresis(fused, width, height, thr * 0.55, thr);

    suppressBorder(mask, width, height, 8); // 모델이 이미지 가장자리(패딩 경계)에서 내는 가짜 선 제거
    removeSpecks(mask, width, height, 30);

    // 모델 선은 ~5px 굵기라 그대로 두면 슬라이더가 "더 굵게"밖에 못 한다. 1px 뼈대로 깎은 뒤
    // 슬라이더만큼 키워서 기본 모드와 같은 감각(0 = 가장 가는 선)으로 맞춘다.
    mask = thinZhangSuen(mask, width, height);
    pruneSpurs(mask, width, height, parseFloat(prune.value));
    const thicknessRadius = parseFloat(thickness.value);
    if (thicknessRadius > 0) mask = dilateDisc(mask, width, height, thicknessRadius);

    return { width, height, mask };
  }

  // ---------- 선화 모드 ----------
  // 검은 윤곽선이 이미 있는 그림: 어두운 픽셀(RGB 최대값 기준, 유채색 어두운 면은 덜 잡히게)을 그대로 선으로.
  // 선의 원래 굵기 변화(붓 터치)를 살리려고 세선화하지 않는다. 눈처럼 검게 칠해진 면도 그대로 검정.
  async function renderInk() {
    // 윤곽선 보충이 켜져 있으면 모델 결과 크기(1024)에 맞춰 선화를 뽑아 합친다
    let outline = null;
    if (outlineAssist.checked) {
      outline = await outlineMaskFromModel();
      if (!outline) return;
    }
    const size = outline ? { width: outline.width, height: outline.height } : null;
    const { width, height, mask } = inkMask(size);
    if (outline) {
      for (let i = 0; i < mask.length; i++) if (outline.mask[i]) mask[i] = 1;
    }
    paintMask(mask, null, width, height);
  }

  // 검은 선·검은 면 마스크. target을 주면 그 크기로, 없으면 기본 크기(긴 변 1200)로.
  function inkMask(target) {
    const { w: sw, h: sh } = sourceSize(activeSource);
    const { width, height } = target || fitSize(sw, sh);
    const src = document.createElement('canvas');
    src.width = width;
    src.height = height;
    const sctx = src.getContext('2d', { willReadFrequently: true });
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(activeSource, 0, 0, width, height);
    const d = sctx.getImageData(0, 0, width, height).data;

    const n = width * height;
    let bright = new Float32Array(n); // RGB 최대값 = "얼마나 안 어두운가"
    for (let i = 0, p = 0; i < d.length; i += 4, p++) bright[p] = Math.max(d[i], d[i + 1], d[i + 2]);

    // 단순화 = JPEG 얼룩·잔 점을 뭉개는 살짝 흐림 (기본 3 → 반경 0.75). 반경 1.5로 했더니 가는 선이
    // 옅어져 기준 아래로 내려가 점선이 됐다(실측). 크게 올리면 가는 선이 사라진다
    const radius = Math.max(0.5, parseFloat(denoise.value)) * 0.25;
    bright = blurFractional(bright, width, height, radius);

    // 어두움 기준 = 얼마나 어두워야 선으로 볼지 (기본 150). 작은 원본을 키우면 안티앨리어싱 때문에
    // 선 픽셀의 최대값이 120~150까지 올라가서 110으로는 점선이 됐다(실측). 회색 그림자가 딸려오면 내릴 것
    const thr = parseFloat(inkDark.value);
    let mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = bright[i] < thr ? 1 : 0;

    // 가장자리 다듬기: 닫기(팽창→침식)로 JPEG 때문에 생긴 핀홀·1px 끊김을 메운다. 반경 1이라 선 굵기는 그대로.
    // 열기(침식→팽창)도 넣어봤는데 1~2px 가는 선을 통째로 지워서 뺐다.
    mask = closeMask(mask, width, height, 1);

    removeSpecks(mask, width, height, 4 + radius * 8);

    // 굵기 1 = 원본 그대로. 크면 그만큼 팽창, 작으면(0~0.5) 그만큼 침식(반전→팽창→반전)
    const t = parseFloat(thickness.value);
    if (t > 1) {
      mask = dilateDisc(mask, width, height, t - 1);
    } else if (t < 1) {
      const inv = new Uint8Array(n);
      for (let i = 0; i < n; i++) inv[i] = mask[i] ? 0 : 1;
      const grown = dilateDisc(inv, width, height, (1 - t) * 2);
      for (let i = 0; i < n; i++) mask[i] = grown[i] ? 0 : 1;
    }

    return { width, height, mask };
  }

  function erodeMask(mask, width, height, radius) {
    const n = width * height;
    const inv = new Uint8Array(n);
    for (let i = 0; i < n; i++) inv[i] = mask[i] ? 0 : 1;
    const grown = dilateDisc(inv, width, height, radius);
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = grown[i] ? 0 : 1;
    return out;
  }
  function closeMask(mask, width, height, radius) {
    return erodeMask(dilateDisc(mask, width, height, radius), width, height, radius);
  }

  // ---------- 인쇄 ----------
  // 브라우저 인쇄로 A4 한 장: 결과물 크게, 원본 우측 하단 작게, 제목 하단 중앙. PDF 저장도 인쇄 대화상자에서.
  function preparePrintSheet() {
    if (!lastPaint) return false;
    printResult.src = resultCanvas.toDataURL('image/png');
    printOriginal.src = originalCanvas.width ? originalCanvas.toDataURL('image/png') : '';
    printTitle.textContent = titleInput.value.trim();
    return true;
  }
  printBtn.addEventListener('click', () => {
    if (!preparePrintSheet()) return;
    window.print();
  });
  window.addEventListener('beforeprint', preparePrintSheet); // Ctrl+P로 눌러도 같은 레이아웃
  window.addEventListener('afterprint', () => {
    printResult.removeAttribute('src'); // 큰 data URL을 붙들고 있지 않도록
    printOriginal.removeAttribute('src');
  });

  // 마스크(1=선) + 음영 마스크(1=연한 회색)를 결과 캔버스에 그린다.
  function paintMask(mask, shadeMask, width, height) {
    const SHADE_TONE = 225;
    resultCanvas.width = width;
    resultCanvas.height = height;
    const out = document.createElement('canvas');
    out.width = width;
    out.height = height;
    const octx = out.getContext('2d');
    const outData = octx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
      let v;
      if (mask[i]) {
        v = 0; // 윤곽선 = 검정
      } else if (shadeMask && shadeMask[i]) {
        v = SHADE_TONE; // 음영 = 연한 회색
      } else {
        v = 255; // 배경 = 흰색
      }
      const o = i * 4;
      outData.data[o] = v;
      outData.data[o + 1] = v;
      outData.data[o + 2] = v;
      outData.data[o + 3] = 255;
    }
    octx.putImageData(outData, 0, 0);
    resultCanvas.getContext('2d').drawImage(out, 0, 0);
    lastPaint = { mask, shadeMask, width, height };
    applyEraseStrokes();
  }

  // ---------- 지우개 ----------
  function applyEraseStrokes() {
    const ctx = resultCanvas.getContext('2d');
    const w = resultCanvas.width;
    for (let s = 0; s < eraseStrokes.length; s++) drawStroke(ctx, eraseStrokes[s], w);
  }

  function drawStroke(ctx, stroke, w) {
    ctx.save();
    ctx.strokeStyle = '#fff';
    ctx.fillStyle = '#fff';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = stroke.r * 2 * w;
    const pts = stroke.pts;
    if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0][0] * w, pts[0][1] * w, stroke.r * w, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * w, pts[0][1] * w);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * w, pts[i][1] * w);
      ctx.stroke();
    }
    ctx.restore();
  }

  // 마지막 결과를 다시 그린 뒤 남은 지우개 자국만 얹는다 (되돌리기/전부 취소)
  function repaintWithStrokes() {
    if (!lastPaint) return;
    const { mask, shadeMask, width, height } = lastPaint;
    paintMask(mask, shadeMask, width, height);
  }

  function updateEraserButtons() {
    undoBtn.disabled = eraseStrokes.length === 0;
    clearEraseBtn.disabled = eraseStrokes.length === 0;
  }

  function setEraser(on) {
    eraserOn = on;
    eraserBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    eraserSizeWrap.hidden = !on;
    resultWrap.classList.toggle('erasing', on);
    if (!on) eraserCursor.hidden = true;
  }

  eraserBtn.addEventListener('click', () => setEraser(!eraserOn));

  undoBtn.addEventListener('click', () => {
    eraseStrokes.pop();
    updateEraserButtons();
    repaintWithStrokes();
  });

  clearEraseBtn.addEventListener('click', () => {
    eraseStrokes = [];
    updateEraserButtons();
    repaintWithStrokes();
  });

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !undoBtn.disabled) {
      e.preventDefault();
      undoBtn.click();
    }
  });

  // 캔버스는 object-fit: contain 으로 틀 안에 레터박스로 놓이므로, 화면 좌표 → 캔버스 비율 좌표로
  // 바꿀 때 실제 그림이 차지하는 영역(스케일·오프셋)을 계산해야 한다. 엘리먼트 사각형만 보면 어긋난다.
  function canvasGeometry() {
    const rect = resultCanvas.getBoundingClientRect();
    const scale = Math.min(rect.width / resultCanvas.width, rect.height / resultCanvas.height);
    const drawnW = resultCanvas.width * scale;
    const drawnH = resultCanvas.height * scale;
    return {
      scale,
      left: rect.left + (rect.width - drawnW) / 2,
      top: rect.top + (rect.height - drawnH) / 2,
      drawnW,
      drawnH,
    };
  }

  function toNormalized(e) {
    const g = canvasGeometry();
    // x, y 모두 "캔버스 폭" 기준 비율로 저장한다 (반지름도 폭 기준) → 가로세로 어디서든 원이 원으로 그려진다
    return [
      (e.clientX - g.left) / g.drawnW,
      (e.clientY - g.top) / g.drawnH * (resultCanvas.height / resultCanvas.width),
    ];
  }

  function moveCursor(e) {
    const g = canvasGeometry();
    const wrapRect = resultWrap.getBoundingClientRect();
    const px = parseFloat(eraserSize.value) * g.scale; // 캔버스 픽셀 → 화면 픽셀
    eraserCursor.style.width = px + 'px';
    eraserCursor.style.height = px + 'px';
    eraserCursor.style.left = (e.clientX - wrapRect.left) + 'px';
    eraserCursor.style.top = (e.clientY - wrapRect.top) + 'px';
    eraserCursor.hidden = false;
  }

  resultWrap.addEventListener('pointerdown', (e) => {
    if (!eraserOn || !lastPaint) return;
    e.preventDefault();
    resultWrap.setPointerCapture(e.pointerId);
    activeStroke = { r: parseFloat(eraserSize.value) / 2 / resultCanvas.width, pts: [toNormalized(e)] };
    eraseStrokes.push(activeStroke);
    drawStroke(resultCanvas.getContext('2d'), activeStroke, resultCanvas.width);
    updateEraserButtons();
    moveCursor(e);
  });

  resultWrap.addEventListener('pointermove', (e) => {
    if (!eraserOn) return;
    moveCursor(e);
    if (!activeStroke) return;
    const p = toNormalized(e);
    const last = activeStroke.pts[activeStroke.pts.length - 1];
    activeStroke.pts.push(p);
    // 마지막 구간만 덧그린다 (전체 다시 그리면 긴 획에서 느려짐)
    const ctx = resultCanvas.getContext('2d');
    const w = resultCanvas.width;
    ctx.save();
    ctx.strokeStyle = '#fff';
    ctx.lineCap = 'round';
    ctx.lineWidth = activeStroke.r * 2 * w;
    ctx.beginPath();
    ctx.moveTo(last[0] * w, last[1] * w);
    ctx.lineTo(p[0] * w, p[1] * w);
    ctx.stroke();
    ctx.restore();
  });

  const endStroke = () => { activeStroke = null; };
  resultWrap.addEventListener('pointerup', endStroke);
  resultWrap.addEventListener('pointercancel', endStroke);
  resultWrap.addEventListener('pointerleave', () => { eraserCursor.hidden = true; });

  // ---------- 이미지 처리 파이프라인 ----------
  //
  // 목표는 "명암을 흑백으로 바꾸기"가 아니라 컬러링북처럼 **이어진 뚜렷한 윤곽선**을 뽑는 것.
  // 밝기 차이에 그냥 임계값을 걸면 점각(speckle)이 흩뿌려지므로, 아래 순서로 처리한다.
  //   1) 블러로 잔 텍스처를 스케일 단위로 지우고
  //   2) 그 스케일에 맞춘 간격으로 밝기 변화(그래디언트)를 재고
  //   3) 변화가 가장 급한 능선 한 줄만 남기고(비최대 억제)
  //   4) 강한 선에서 이어지는 약한 선만 살려 끊긴 점을 버리고(히스테리시스)
  //   5) 그래도 남은 작은 얼룩은 덩어리 크기로 제거한 뒤
  //   6) 원하는 굵기로 두껍게 만든다.
  function render() {
    const { w: sw, h: sh } = sourceSize(activeSource);
    const { width, height } = fitSize(sw, sh);

    const src = document.createElement('canvas');
    src.width = width;
    src.height = height;
    const sctx = src.getContext('2d', { willReadFrequently: true });
    sctx.imageSmoothingQuality = 'high'; // 확대 시 계단 현상 대신 부드러운 경사로 → 엣지 검출이 깔끔
    sctx.drawImage(activeSource, 0, 0, width, height);
    const imageData = sctx.getImageData(0, 0, width, height);

    // 흑백(휘도)만 보면 캐릭터 그림에서 중요한 경계가 통째로 사라진다:
    //   흰 배경 위 노랑(피카츄) = 휘도 차이 ~50 뿐이라 얼굴 윤곽이 끊기고,
    //   노랑 위 빨강 볼·분홍 입은 휘도로는 흐릿한데 색으로는 확연하다.
    // 그래서 R/G/B 채널을 따로 흐린 뒤 세 채널의 변화를 합쳐 "색이 바뀌는 곳"을 선으로 잡는다.
    const channels = splitChannels(imageData);

    // 1) 단순화: 반경이 클수록 잔 디테일이 사라져 아이들용 단순한 그림이 된다.
    //    박스 블러 2회는 가우시안 블러의 값싼 근사. 반경은 0.5 단위(소수)까지 받는다.
    const radius = Math.max(0.5, parseFloat(denoise.value));
    const blur2 = (ch) => blurFractional(blurFractional(ch, width, height, radius), width, height, radius);
    const smoothRGB = [blur2(channels.r), blur2(channels.g), blur2(channels.b)];

    // 2) 블러 반경만큼 떨어진 픽셀끼리 비교한다. 블러가 경계를 폭 ~2*radius의 완만한
    //    경사로 펴놓기 때문에, 이웃 1픽셀만 보는 고정 3x3 커널로는 그 경사를 못 잡는다.
    const step = Math.max(1, Math.round(radius));
    const { mag, dir } = gradientColor(smoothRGB, width, height, step);

    // 3) 경사면 전체가 아니라 능선(정점)만 남긴다 — 굵은 띠가 아니라 한 줄 선이 되도록.
    const ridge = nonMaxSuppress(mag, dir, width, height, step);

    // 4) 영역별 적응: 사진 전체에 기준 하나만 쓰면, 대비가 극단적으로 센 부분(검은 용 vs 금색 배경)이
    //    상위권을 독식해서 대비가 약한 부분(밝은 것 위의 밝은 것 = 해·광배)이 통째로 잘려나간다.
    //    그래서 픽셀마다 "자기 주변 동네의 평균 대비"로 나눠 상대적 세기로 바꾼다.
    //    격자로 자르면 칸 경계에서 선이 끊기므로, 창이 미끄러지는 방식(박스 블러)을 쓴다.
    const score = adaptiveScore(ridge, mag, width, height, parseFloat(adaptive.value) / 100);

    // 5) 임계값을 절대값으로 고정하면 사진마다 결과가 들쭉날쭉하다.
    //    이 사진 자신의 분포에서 "전체 픽셀의 상위 몇 %를 선의 씨앗으로 쓸지"로 정해 일관성을 준다.
    //    지수 곡선이라 낮은 쪽(정교한 조절이 필요한 구간)이 촘촘하다.
    //    슬라이더는 115까지 열어뒀다. 곡선은 그대로라 100 이하 눈금의 의미는 변하지 않고
    //    (기존에 맞춰둔 세팅이 그대로 재현된다) 100 너머만 더 촘촘한 구간으로 확장된다.
    //    130까지 열어봤더니 화면의 98%가 검게 차버려서(측정) 쓸모없는 구간은 잘라냈다.
    const sensitivityPct = parseFloat(sensitivity.value);
    const seedPct = 0.05 * Math.pow(100, sensitivityPct / 100); // 0.05%(0) ~ 5%(100) ~ 10%(115)
    const high = percentile(score, seedPct);
    const low = high * 0.5;
    // 강한 선에서 출발해 이어지는 약한 선만 살린다 → 흩어진 점이 아니라 이어진 윤곽선.
    let mask = hysteresis(score, width, height, low, high);

    // 블러는 이미지 바깥을 가장자리 픽셀 복제로 채우므로 테두리에 가짜 선이 생긴다.
    suppressBorder(mask, width, height, Math.ceil(radius) * 2);

    // 6) 남은 점각 제거 — 단순화를 올릴수록 더 큰 덩어리만 남긴다.
    removeSpecks(mask, width, height, 4 + radius * 8);
    pruneSpurs(mask, width, height, parseFloat(prune.value));

    // 7) 굵기
    const thicknessRadius = parseFloat(thickness.value);
    if (thicknessRadius > 0) {
      mask = dilateDisc(mask, width, height, thicknessRadius);
    }

    // 음영(옵션): 어두운 영역에 연한 회색 플랫톤만 깐다. 해칭 노이즈가 아니라 면 단위 채우기.
    let shadeMask = null;
    if (shading.checked) {
      const smooth = blur2(channels.gray); // 음영은 밝기 기준이라 휘도를 따로 흐린다
      let sum = 0;
      for (let i = 0; i < smooth.length; i++) sum += smooth[i];
      const shadeThreshold = (sum / smooth.length) * 0.65;
      shadeMask = new Uint8Array(smooth.length);
      for (let i = 0; i < smooth.length; i++) {
        shadeMask[i] = smooth[i] < shadeThreshold ? 1 : 0;
      }
    }

    paintMask(mask, shadeMask, width, height);
  }

  // R/G/B 채널과 휘도를 각각 Float32 배열로 분리
  function splitChannels(imageData) {
    const { data, width, height } = imageData;
    const n = width * height;
    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    const gray = new Float32Array(n);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      r[p] = data[i];
      g[p] = data[i + 1];
      b[p] = data[i + 2];
      gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return { r, g, b, gray };
  }

  // 페이지가 뜨자마자 AI 라이브러리+모델을 미리 받아둔다 (0.4MB 모델 + 런타임). 실패해도 조용히 —
  // 실제 변환 때 다시 시도하고 그때 에러를 보여준다.
  getAISession().catch(() => {});
  getBackgroundSession().catch(() => {});

  function toGrayscale(imageData) {
    const { data, width, height } = imageData;
    const gray = new Float32Array(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // 표준 휘도 가중치
      gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return gray;
  }

  // 분리형 박스 블러 (수평 -> 수직)
  function boxBlur(src, width, height, radius) {
    const tmp = new Float32Array(width * height);
    const out = new Float32Array(width * height);
    const size = radius * 2 + 1;

    for (let y = 0; y < height; y++) {
      const rowOff = y * width;
      let sum = 0;
      for (let x = -radius; x <= radius; x++) {
        sum += src[rowOff + clamp(x, 0, width - 1)];
      }
      for (let x = 0; x < width; x++) {
        tmp[rowOff + x] = sum / size;
        const addX = clamp(x + radius + 1, 0, width - 1);
        const subX = clamp(x - radius, 0, width - 1);
        sum += src[rowOff + addX] - src[rowOff + subX];
      }
    }

    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let y = -radius; y <= radius; y++) {
        sum += tmp[clamp(y, 0, height - 1) * width + x];
      }
      for (let y = 0; y < height; y++) {
        out[y * width + x] = sum / size;
        const addY = clamp(y + radius + 1, 0, height - 1);
        const subY = clamp(y - radius, 0, height - 1);
        sum += tmp[addY * width + x] - tmp[subY * width + x];
      }
    }
    return out;
  }

  // 소수 반경 블러: 정수 반경 두 개를 섞어 0.5 같은 중간값을 만든다.
  // (박스 블러는 정수 반경만 되는데, 단순화 슬라이더를 촘촘하게 쓰려면 중간값이 필요하다)
  function blurFractional(src, width, height, radius) {
    const lo = Math.floor(radius);
    const frac = radius - lo;
    const a = boxBlur(src, width, height, lo);
    if (frac === 0) return a;
    const b = boxBlur(src, width, height, lo + 1);
    for (let i = 0; i < a.length; i++) a[i] += (b[i] - a[i]) * frac;
    return a;
  }

  // 픽셀마다 "자기 주변 동네의 평균 대비"로 나눠 상대적 세기로 바꾼다.
  // strength=0 이면 사진 전체 평균만 쓰므로 적응 전과 완전히 동일한 결과가 나온다(안전한 원위치).
  function adaptiveScore(ridge, mag, width, height, strength) {
    let sum = 0;
    for (let i = 0; i < mag.length; i++) sum += mag[i];
    const globalMean = sum / mag.length || 1;

    // 전체 픽셀 수에 비례하는 크기의 창. 무시하고 싶은 텍스처보다는 크고,
    // 적응하고 싶은 밝기 변화(해 주변 vs 어두운 바위)보다는 작아야 한다.
    const window = Math.max(8, Math.round(Math.min(width, height) / 8));
    const localMean = boxBlur(mag, width, height, window);

    // 안전장치(하한선): 아무것도 없는 평평한 하늘에서 동네 평균이 0에 가까워지면
    // 미세한 노이즈가 "이 동네 1등"이 되어 없는 선이 생긴다. 그래서 바닥을 깔아둔다.
    const floor = globalMean * 0.35;

    const score = new Float32Array(ridge.length);
    for (let i = 0; i < ridge.length; i++) {
      if (ridge[i] === 0) continue;
      const local = localMean[i] > floor ? localMean[i] : floor;
      const denom = globalMean + (local - globalMean) * strength;
      score[i] = ridge[i] / denom;
    }
    return score;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // 색 변화의 세기와 방향. 세 채널의 변화 벡터를 합쳐(RGB 공간에서의 거리) 세기로 쓰고,
  // 방향은 가장 크게 변한 채널의 것을 따른다 — 방향은 NMS에서 4방향으로 양자화되므로 이 정도로 충분.
  // dir: 0=가로변화(│선) 1=대각(/) 2=세로변화(─선) 3=대각(\)
  function gradientColor(chs, width, height, step) {
    const mag = new Float32Array(width * height);
    const dir = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      const rowT = clamp(y - step, 0, height - 1) * width;
      const rowB = clamp(y + step, 0, height - 1) * width;
      for (let x = 0; x < width; x++) {
        const xl = row + clamp(x - step, 0, width - 1);
        const xr = row + clamp(x + step, 0, width - 1);
        let sum = 0;
        let best = -1;
        let bgx = 0;
        let bgy = 0;
        for (let c = 0; c < chs.length; c++) {
          const ch = chs[c];
          const gx = ch[xr] - ch[xl];
          const gy = ch[rowB + x] - ch[rowT + x];
          const m = gx * gx + gy * gy;
          sum += m;
          if (m > best) { best = m; bgx = gx; bgy = gy; }
        }
        mag[row + x] = Math.sqrt(sum);
        const angle = ((Math.atan2(bgy, bgx) * 180 / Math.PI) % 180 + 180) % 180;
        dir[row + x] = (angle < 22.5 || angle >= 157.5) ? 0 : angle < 67.5 ? 1 : angle < 112.5 ? 2 : 3;
      }
    }
    return { mag, dir };
  }

  // 밝기 변화의 세기와 방향. step은 블러 반경에 맞춘다.
  // dir: 0=가로변화(│선) 1=대각(/) 2=세로변화(─선) 3=대각(\)
  function gradient(gray, width, height, step) {
    const mag = new Float32Array(width * height);
    const dir = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      const rowT = clamp(y - step, 0, height - 1) * width;
      const rowB = clamp(y + step, 0, height - 1) * width;
      for (let x = 0; x < width; x++) {
        const gx = gray[row + clamp(x + step, 0, width - 1)] - gray[row + clamp(x - step, 0, width - 1)];
        const gy = gray[rowB + x] - gray[rowT + x];
        mag[row + x] = Math.sqrt(gx * gx + gy * gy);
        // 0~180도로 접어 4방향으로 양자화
        const angle = ((Math.atan2(gy, gx) * 180 / Math.PI) % 180 + 180) % 180;
        dir[row + x] = (angle < 22.5 || angle >= 157.5) ? 0 : angle < 67.5 ? 1 : angle < 112.5 ? 2 : 3;
      }
    }
    return { mag, dir };
  }

  // 그래디언트 방향으로 ±step 떨어진 두 이웃보다 크지 않으면 버린다 → 능선 한 줄만 남음.
  function nonMaxSuppress(mag, dir, width, height, step) {
    const out = new Float32Array(mag.length);
    const offX = [step, step, 0, step];
    const offY = [0, step, step, -step];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const v = mag[i];
        if (v === 0) continue;
        const d = dir[i];
        const ax = offX[d];
        const ay = offY[d];
        const a = clamp(y + ay, 0, height - 1) * width + clamp(x + ax, 0, width - 1);
        const b = clamp(y - ay, 0, height - 1) * width + clamp(x - ax, 0, width - 1);
        if (v >= mag[a] && v >= mag[b]) out[i] = v;
      }
    }
    return out;
  }

  // 전체 픽셀 중 상위 keepPct% 지점의 값을 구한다(히스토그램 근사).
  //
  // 칸 수가 적으면 선을 많이 뽑는 구간(=임계값이 낮은 구간)에서 한 칸에 픽셀이 수만 개씩 몰려,
  // 슬라이더를 움직여도 같은 칸에 머물러 결과가 전혀 안 변한다(선 개수 100과 105가 동일했던 원인).
  // 칸을 늘리고, 마지막 칸 안에서 선형 보간까지 해서 연속적으로 움직이게 한다.
  function percentile(values, keepPct) {
    const BINS = 4096;
    let max = 0;
    for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
    if (max <= 0) return Infinity;

    const scale = (BINS - 1) / max;
    const hist = new Int32Array(BINS);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v > 0) hist[(v * scale) | 0]++;
    }

    const keep = Math.max(1, values.length * keepPct / 100);
    let acc = 0;
    for (let b = BINS - 1; b >= 0; b--) {
      const count = hist[b];
      if (acc + count >= keep) {
        // 이 칸 안에서 위쪽부터 채운다고 보고 보간 (칸 경계에서 계단처럼 튀지 않도록)
        const frac = count > 0 ? (keep - acc) / count : 0;
        return (b + 1 - frac) / scale;
      }
      acc += count;
    }
    return 0;
  }

  // high 이상인 픽셀을 씨앗으로 삼아, 8방향으로 이어지는 low 이상 픽셀까지 따라가며 선을 잇는다.
  // 강한 선에 닿지 못한 약한 응답(=흩어진 점각)은 자동으로 탈락한다.
  function hysteresis(mag, width, height, low, high) {
    const mask = new Uint8Array(mag.length);
    const stack = new Int32Array(mag.length);
    let top = 0;

    for (let i = 0; i < mag.length; i++) {
      if (mag[i] >= high) {
        mask[i] = 1;
        stack[top++] = i;
      }
    }

    while (top > 0) {
      const i = stack[--top];
      const x = i % width;
      const y = (i / width) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const j = ny * width + nx;
          if (!mask[j] && mag[j] >= low) {
            mask[j] = 1;
            stack[top++] = j;
          }
        }
      }
    }
    return mask;
  }

  // minSize보다 작은 덩어리(연결 성분)를 지운다.
  function removeSpecks(mask, width, height, minSize) {
    if (minSize <= 1) return;
    const seen = new Uint8Array(mask.length);
    const stack = new Int32Array(mask.length);
    const comp = new Int32Array(mask.length);

    for (let s = 0; s < mask.length; s++) {
      if (!mask[s] || seen[s]) continue;
      let top = 0;
      let n = 0;
      stack[top++] = s;
      seen[s] = 1;
      while (top > 0) {
        const i = stack[--top];
        comp[n++] = i;
        const x = i % width;
        const y = (i / width) | 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            const j = ny * width + nx;
            if (mask[j] && !seen[j]) {
              seen[j] = 1;
              stack[top++] = j;
            }
          }
        }
      }
      if (n < minSize) {
        for (let k = 0; k < n; k++) mask[comp[k]] = 0;
      }
    }
  }

  function suppressBorder(mask, width, height, margin) {
    const m = Math.min(margin, Math.floor(Math.min(width, height) / 2));
    if (m <= 0) return;
    for (let y = 0; y < height; y++) {
      const inBand = y < m || y >= height - m;
      const row = y * width;
      for (let x = 0; x < width; x++) {
        if (inBand || x < m || x >= width - m) mask[row + x] = 0;
      }
    }
  }

  // 원형 커널 최대값 필터 (선 굵기 확장).
  // 정사각 커널은 선이 각지게 굵어지고 크기가 껑충 뛰어서(9→25→49px) 중간 굵기가 없다.
  // 원형은 면적이 완만하게 늘어 0.5 단위 조절이 실제로 눈에 보인다.
  // 가지치기: 1px 선에서 "끝점 → 분기점(또는 다른 끝점)"까지 길이가 maxLen 이하인 가지를 지운다.
  // 진짜 윤곽은 닫힌 고리라 끝점이 없고, 봉제선·털 주름·JPEG 노이즈는 끝이 뚫린 짧은 획이다.
  // 입·눈썹처럼 열려 있지만 긴 획은 maxLen보다 길어 살아남는다 (1024px 기준 기본 30).
  function pruneSpurs(mask, width, height, maxLen) {
    const steps = Math.round(maxLen);
    if (steps <= 0) return;
    const n = width * height;
    const countNeighbors = (m, i) => {
      const x = i % width;
      const y = (i / width) | 0;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if (m[ny * width + nx]) k++;
        }
      }
      return k;
    };
    const orig = mask.slice();

    // 0) 고립된 열린 획(입·눈썹처럼 분기점이 하나도 없는 덩어리)은 "특징"으로 보고 보호한다.
    //    가지치기는 윤곽에 매달린 가지(봉제선·주름)만 대상. 단, 아주 짧은 고립 획(12px 미만)은 노이즈로 지운다.
    const protectedPx = new Uint8Array(n);
    {
      const label = new Int32Array(n); // 0 = 미방문
      const stack = new Int32Array(n);
      let compId = 0;
      for (let seed = 0; seed < n; seed++) {
        if (!mask[seed] || label[seed]) continue;
        compId++;
        let top = 0;
        stack[top++] = seed;
        label[seed] = compId;
        const members = [];
        let hasJunction = false;
        while (top > 0) {
          const i = stack[--top];
          members.push(i);
          const k = countNeighbors(mask, i);
          if (k >= 3) hasJunction = true;
          const x = i % width;
          const y = (i / width) | 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;
              const nx = x + dx, ny = y + dy;
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
              const j = ny * width + nx;
              if (mask[j] && !label[j]) { label[j] = compId; stack[top++] = j; }
            }
          }
        }
        if (!hasJunction) {
          if (members.length < 12) for (const i of members) mask[i] = 0;
          else for (const i of members) protectedPx[i] = 1;
        }
      }
    }

    // 1) 끝점(이웃 1개 이하)을 steps번 깎는다 → 길이 steps 이하인 가지는 통째로 사라지고,
    //    긴 열린 획은 양끝이 steps만큼 짧아진다.
    let frontier = [];
    for (let i = 0; i < n; i++) if (mask[i] && !protectedPx[i] && countNeighbors(mask, i) <= 1) frontier.push(i);
    for (let s = 0; s < steps && frontier.length; s++) {
      const next = [];
      for (let t = 0; t < frontier.length; t++) mask[frontier[t]] = 0;
      for (let t = 0; t < frontier.length; t++) {
        const i = frontier[t];
        const x = i % width;
        const y = (i / width) | 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const j = ny * width + nx;
            if (mask[j] && !protectedPx[j] && countNeighbors(mask, j) <= 1 && next.indexOf(j) < 0) next.push(j);
          }
        }
      }
      frontier = next;
    }

    // 2) 살아남은 끝점에서만 원래 선을 따라 steps번 되살린다 → 짧아졌던 긴 획은 복구되고,
    //    분기점에 붙어 있다가 지워진 가지는 (분기점은 끝점이 아니므로) 되살아나지 않는다.
    frontier = [];
    for (let i = 0; i < n; i++) if (mask[i] && countNeighbors(mask, i) === 1) frontier.push(i);
    for (let s = 0; s < steps && frontier.length; s++) {
      const next = [];
      for (let t = 0; t < frontier.length; t++) {
        const i = frontier[t];
        const x = i % width;
        const y = (i / width) | 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const j = ny * width + nx;
            if (orig[j] && !mask[j]) { mask[j] = 1; next.push(j); }
          }
        }
      }
      frontier = next;
    }
  }

  // Zhang-Suen 세선화: 굵은 선을 가운데 1px 뼈대만 남기고 깎는다. 연결은 끊지 않는다.
  // 두 패스를 번갈아 돌리며 더 지울 픽셀이 없을 때까지 반복 (선 굵기 5px면 3번 안팎).
  function thinZhangSuen(mask, width, height) {
    const toDelete = [];
    let changed = true;
    while (changed) {
      changed = false;
      for (let pass = 0; pass < 2; pass++) {
        toDelete.length = 0;
        for (let y = 1; y < height - 1; y++) {
          for (let x = 1; x < width - 1; x++) {
            const i = y * width + x;
            if (!mask[i]) continue;
            // 8이웃을 시계 방향으로 p2(위)부터
            const p2 = mask[i - width], p3 = mask[i - width + 1], p4 = mask[i + 1], p5 = mask[i + width + 1];
            const p6 = mask[i + width], p7 = mask[i + width - 1], p8 = mask[i - 1], p9 = mask[i - width - 1];
            const b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            if (b < 2 || b > 6) continue; // 끝점이거나 안쪽 픽셀이면 유지
            // 0→1 전이가 정확히 한 번이어야 지워도 연결이 안 끊긴다
            let a = 0;
            if (!p2 && p3) a++; if (!p3 && p4) a++; if (!p4 && p5) a++; if (!p5 && p6) a++;
            if (!p6 && p7) a++; if (!p7 && p8) a++; if (!p8 && p9) a++; if (!p9 && p2) a++;
            if (a !== 1) continue;
            if (pass === 0) {
              if ((p2 && p4 && p6) || (p4 && p6 && p8)) continue;
            } else if ((p2 && p4 && p8) || (p2 && p6 && p8)) {
              continue;
            }
            toDelete.push(i);
          }
        }
        for (let k = 0; k < toDelete.length; k++) mask[toDelete[k]] = 0;
        if (toDelete.length) changed = true;
      }
    }
    return mask;
  }

  function dilateDisc(mask, width, height, radius) {
    const r = Math.ceil(radius);
    const rSq = radius * radius;
    const offX = [];
    const offY = [];
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy <= rSq) { offX.push(dx); offY.push(dy); }
      }
    }

    const out = new Uint8Array(mask.length);
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;
      const x = i % width;
      const y = (i / width) | 0;
      for (let k = 0; k < offX.length; k++) {
        const nx = x + offX[k];
        const ny = y + offY[k];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        out[ny * width + nx] = 1;
      }
    }
    return out;
  }
})();
