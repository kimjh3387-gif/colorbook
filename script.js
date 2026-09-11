// 색칠공부 변환기 — 모든 처리는 브라우저 안에서만 일어난다 (업로드/네트워크 요청 없음)
//
// v4: "선화" 하나로 통합. 원본에 있는 것만 그대로 뽑는다 — 없는 선을 "그리는" 방식(학습된 엣지 모델)은
// 선이 울퉁불퉁하고 봉제선까지 끌어와서 뺐다(사용자 실측 결론). 세 층을 합친다:
//   1) 검은 선·검은 면  : 어두운 픽셀 그대로 (만화·일러스트의 윤곽, 눈동자, 귀 끝)
//   2) 실루엣 선        : 배경 분리 마스크의 바깥 테두리 (검은 선이 없는 인형·사진의 몸통 윤곽)
//   3) 색 경계 선       : 색이 확 바뀌는 곳 (볼·무늬). 그림자 같은 애매한 경계는 무시
// 2·3은 이미 검은 선이 있는 자리엔 겹치지 않게 한다(이중선 방지).
(() => {
  const MAX_DIMENSION = 1600; // 성능 보호용 최대 변 길이
  // 작은 원본(이미지 검색 썸네일을 우클릭 저장한 400~600px짜리가 대부분)은 처리 전에 이 크기로 확대한다.
  // 결과 PNG도 인쇄하기엔 너무 작고, 확대해 두면 슬라이더 감각이 큰 사진과 같아진다.
  const MIN_DIMENSION = 1200;

  // ---------- 배경 제거 (U²-Net-p) ----------
  // 숲 배경 애니 장면처럼 배경이 있으면 배경 텍스처가 선으로 딸려온다. 그래서 변환 전에 주요 피사체만
  // 남기고 배경을 흰색으로 밀어낸다. U²-Net-p(4.6MB, Apache-2.0 → 광고 붙여도 됨)를 320×320으로 돌려
  // 마스크를 얻고 원본 크기로 키워 합성. 같은 마스크의 테두리가 "실루엣 선"이 된다.
  // 실측 CPU 150~190ms. 한계: 경량 모델이라 "주인공 하나" 위주 — 옆의 두 번째 캐릭터는 같이 지워질 수 있다.
  // 함정: 원본 ONNX의 MaxPool ceil_mode=1을 WebGPU가 거부해서 0으로 바꿔 저장했다(models/README.md).
  const BG_MODEL_URL = 'models/u2netp.onnx';
  const ORT_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js';
  const BG_INPUT = 320;
  const BG_MEAN = [0.485, 0.456, 0.406]; // ImageNet 정규화 (모델 학습 조건)
  const BG_STD = [0.229, 0.224, 0.225];

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

  const bgRemove = document.getElementById('bgRemove');
  const silhouette = document.getElementById('silhouette');
  const colorEdge = document.getElementById('colorEdge');
  const colorEdgeStrength = document.getElementById('colorEdgeStrength');
  const colorEdgeStrengthVal = document.getElementById('colorEdgeStrengthVal');
  const inkDark = document.getElementById('inkDark');
  const inkDarkVal = document.getElementById('inkDarkVal');
  const denoise = document.getElementById('denoise');
  const denoiseVal = document.getElementById('denoiseVal');
  const thickness = document.getElementById('thickness');
  const thicknessVal = document.getElementById('thicknessVal');

  const titleInput = document.getElementById('titleInput');
  const printBtn = document.getElementById('printBtn');
  const printResult = document.getElementById('printResult');
  const printOriginal = document.getElementById('printOriginal');
  const printTitle = document.getElementById('printTitle');

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

  // inkDark 150: 작은 원본을 1200으로 키우면 안티앨리어싱 때문에 선 픽셀 최대값이 120~150까지 올라가서
  // 110으로는 점선이 됐다(실측). 회색 그림자가 딸려오면 내릴 것.
  const DEFAULTS = {
    inkDark: 150, denoise: 3, thickness: 1, colorEdgeStrength: 70,
    bgRemove: true, silhouette: true, colorEdge: true,
  };

  let sourceImage = null;      // HTMLImageElement (원본)
  let activeSource = null;     // 이번 렌더가 실제로 읽는 소스 (원본 이미지 또는 배경 지운 캔버스)
  let rendering = false;       // 렌더가 진행 중 (배경 모델을 await 하는 동안 입력 이벤트가 계속 들어온다)
  let rerunRequested = false;  // 진행 중에 슬라이더가 움직였으면 끝나고 한 번 더
  let bgSessionPromise = null; // onnxruntime 세션 (한 번만 만든다)
  let bgCache = null;          // { image, canvas, maskCanvas } — 원본당 한 번만 배경을 지운다

  // ---------- 지우개 상태 ----------
  // 알고리즘이 못 가리는 선(인형 봉제선 등)은 사람이 지운다. 지운 자국은 캔버스 폭에 대한 비율 좌표로
  // 저장해서, 슬라이더를 다시 만져 결과가 새로 그려져도(=paintMask) 같은 자리에 다시 적용된다. 새 이미지를 올리면 비운다.
  let eraseStrokes = [];   // [{ r: 반지름(폭 대비 비율), pts: [[x,y], ...] }]
  let lastPaint = null;    // 마지막으로 그린 { mask, width, height } — 되돌리기용 재도색
  let eraserOn = false;
  let activeStroke = null;

  // 소스가 <img>든 <canvas>든 같은 방식으로 크기를 읽는다
  function sourceSize(src) {
    return src instanceof HTMLCanvasElement
      ? { w: src.width, h: src.height }
      : { w: src.naturalWidth, h: src.naturalHeight };
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
    [inkDark, inkDarkVal],
    [denoise, denoiseVal],
    [thickness, thicknessVal],
    [colorEdgeStrength, colorEdgeStrengthVal],
  ].forEach(([input, out]) => {
    input.addEventListener('input', () => {
      out.textContent = input.value;
      scheduleRender();
    });
  });
  [bgRemove, silhouette, colorEdge].forEach((box) => box.addEventListener('change', scheduleRender));

  resetBtn.addEventListener('click', () => {
    inkDark.value = DEFAULTS.inkDark;
    denoise.value = DEFAULTS.denoise;
    thickness.value = DEFAULTS.thickness;
    colorEdgeStrength.value = DEFAULTS.colorEdgeStrength;
    inkDarkVal.textContent = DEFAULTS.inkDark;
    denoiseVal.textContent = DEFAULTS.denoise;
    thicknessVal.textContent = DEFAULTS.thickness;
    colorEdgeStrengthVal.textContent = DEFAULTS.colorEdgeStrength;
    bgRemove.checked = DEFAULTS.bgRemove;
    silhouette.checked = DEFAULTS.silhouette;
    colorEdge.checked = DEFAULTS.colorEdge;
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
  // (배경 모델을 await 하는 동안 슬라이더 이벤트가 진짜로 들어오기 때문에 "큐에 있으면 무시"면 마지막 위치가 버려진다)
  function scheduleRender() {
    if (!sourceImage) return;
    rerunRequested = true;
    if (rendering) return;
    rendering = true;
    processingOverlay.textContent = '변환 중…';
    processingOverlay.hidden = false;
    // requestAnimationFrame에 의존하면 탭이 백그라운드/비표시 상태일 때 콜백이 멈춘다. setTimeout만 쓴다.
    setTimeout(async () => {
      try {
        while (rerunRequested) {
          rerunRequested = false;
          // 배경 지우기든 실루엣 선이든 마스크가 필요하면 모델을 돌린다 (원본당 한 번, 캐시)
          const info = (bgRemove.checked || silhouette.checked) ? await getBackgroundInfo() : null;
          if (rerunRequested) continue; // 기다리는 사이 입력이 바뀜 — 처음부터
          activeSource = (bgRemove.checked && info) ? info.canvas : sourceImage;
          processingOverlay.textContent = '변환 중…';
          render(info);
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

  // ---------- 배경 제거 ----------
  function loadScript(url) {
    return new Promise((resolve, reject) => {
      if (window.ort) { resolve(); return; }
      const el = document.createElement('script');
      el.src = url;
      el.onload = resolve;
      el.onerror = () => reject(new Error('배경 분리 라이브러리를 못 불러왔어요 (인터넷 연결 확인)'));
      document.head.appendChild(el);
    });
  }

  function getBackgroundSession() {
    if (bgSessionPromise) return bgSessionPromise;
    bgSessionPromise = (async () => {
      await loadScript(ORT_SCRIPT_URL);
      // WebGPU가 되면 GPU, 아니면 WASM으로 자동 폴백
      return ort.InferenceSession.create(BG_MODEL_URL, { executionProviders: ['webgpu', 'wasm'] });
    })();
    bgSessionPromise.catch(() => { bgSessionPromise = null; }); // 실패하면 다음에 다시 시도할 수 있게
    return bgSessionPromise;
  }

  // 원본 → { canvas: 배경을 흰색으로 밀어낸 캔버스, maskCanvas: 피사체 마스크(회색, 원본 크기) }. 원본당 한 번만.
  async function getBackgroundInfo() {
    if (bgCache && bgCache.image === sourceImage) return bgCache;
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
        x[ch * n + i] = (d[i * 4 + ch] / 255 - BG_MEAN[ch]) / BG_STD[ch];
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
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = W;
    maskCanvas.height = H;
    const bctx = maskCanvas.getContext('2d', { willReadFrequently: true });
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

    bgCache = { image, canvas: outCanvas, maskCanvas };
    return bgCache;
  }

  // 페이지가 뜨자마자 라이브러리+모델을 미리 받아둔다. 실패해도 조용히 — 실제 변환 때 다시 시도하고 그때 에러를 보여준다.
  getBackgroundSession().catch(() => {});

  // ---------- 변환 ----------
  function render(info) {
    const { w: sw, h: sh } = sourceSize(activeSource);
    const { width, height } = fitSize(sw, sh);
    const n = width * height;

    // 소스를 작업 크기로 한 번만 읽는다 (검은 선·색 경계가 같이 쓴다)
    const src = document.createElement('canvas');
    src.width = width;
    src.height = height;
    const sctx = src.getContext('2d', { willReadFrequently: true });
    sctx.imageSmoothingQuality = 'high'; // 확대 시 계단 현상 대신 부드러운 경사로
    sctx.drawImage(activeSource, 0, 0, width, height);
    const imageData = sctx.getImageData(0, 0, width, height);

    // 1) 검은 선·검은 면
    let mask = inkMask(imageData, width, height);

    // 2·3은 이미 검은 선이 있는 자리(6px 이내)엔 얹지 않는다 — 일러스트에서 윤곽선 바깥에 실루엣이
    //    한 줄 더 생기거나(이중선), 검은 선 양옆에 색 경계가 따라붙는 걸 막는다.
    const nearInk = dilateDisc(mask, width, height, 6);
    const addLayer = (layer) => {
      for (let i = 0; i < n; i++) if (layer[i] && !nearInk[i]) mask[i] = 1;
    };

    // 2) 실루엣 선 (배경 분리 마스크의 테두리)
    if (silhouette.checked && info) addLayer(silhouetteMask(info.maskCanvas, width, height));

    // 3) 색 경계 선
    if (colorEdge.checked) addLayer(colorEdgeMask(imageData, width, height, parseFloat(colorEdgeStrength.value)));

    // 굵기 1 = 원본 그대로. 크면 그만큼 팽창, 작으면(0~0.5) 그만큼 침식
    const t = parseFloat(thickness.value);
    if (t > 1) mask = dilateDisc(mask, width, height, t - 1);
    else if (t < 1) mask = erodeMask(mask, width, height, (1 - t) * 2);

    paintMask(mask, width, height);
  }

  // 검은 선·검은 면: 어두운 픽셀(RGB 최대값 기준 — 유채색 어두운 면은 덜 잡히게)을 그대로.
  // 선의 원래 굵기 변화(붓 터치)를 살리려고 세선화하지 않는다. 눈처럼 검게 칠해진 면도 그대로 검정.
  function inkMask(imageData, width, height) {
    const d = imageData.data;
    const n = width * height;
    let bright = new Float32Array(n); // RGB 최대값 = "얼마나 안 어두운가"
    for (let i = 0, p = 0; i < d.length; i += 4, p++) bright[p] = Math.max(d[i], d[i + 1], d[i + 2]);

    // 단순화 = JPEG 얼룩·잔 점을 뭉개는 살짝 흐림 (기본 3 → 반경 0.75). 반경 1.5로 했더니 가는 선이
    // 옅어져 기준 아래로 내려가 점선이 됐다(실측). 크게 올리면 가는 선이 사라진다
    const radius = Math.max(0.5, parseFloat(denoise.value)) * 0.25;
    bright = blurFractional(bright, width, height, radius);

    // 어두움 기준 = 얼마나 어두워야 선으로 볼지 (기본 150). 진한 갈색·남색 윤곽선(최대값 ~90)도 걸린다
    const thr = parseFloat(inkDark.value);
    let mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = bright[i] < thr ? 1 : 0;

    // 닫기(팽창→침식, 반경 1)로 JPEG 때문에 생긴 핀홀·1px 끊김을 메운다. 선 굵기는 그대로.
    // 열기(침식→팽창)도 넣어봤는데 1~2px 가는 선을 통째로 지워서 뺐다.
    mask = closeMask(mask, width, height, 1);

    removeSpecks(mask, width, height, 4 + radius * 8);
    return mask;
  }

  // 실루엣 선: 배경 분리 마스크(피사체=밝음)를 작업 크기로 키워 테두리 픽셀만 남기고 ~4px로 굵힌다.
  // 모델 마스크는 매끈해서 AI 엣지 검출처럼 흔들리지 않는다. 캐릭터가 이미지 가장자리에 닿아 있으면
  // 가장자리를 따라 선이 생기므로 테두리 3px은 지운다.
  function silhouetteMask(maskCanvas, width, height) {
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(maskCanvas, 0, 0, width, height);
    const d = ctx.getImageData(0, 0, width, height).data;
    const n = width * height;
    const fg = new Uint8Array(n);
    for (let i = 0; i < n; i++) fg[i] = d[i * 4] >= 128 ? 1 : 0;

    // 살짝 다듬어서 마스크 가장자리의 계단(320→1200 확대)을 편다
    let smooth = closeMask(fg, width, height, 2);
    smooth = erodeMask(dilateDisc(smooth, width, height, 2), width, height, 2);

    const edge = new Uint8Array(n);
    for (let y = 1; y < height - 1; y++) {
      const row = y * width;
      for (let x = 1; x < width - 1; x++) {
        const i = row + x;
        if (!smooth[i]) continue;
        if (!smooth[i - 1] || !smooth[i + 1] || !smooth[i - width] || !smooth[i + width]) edge[i] = 1;
      }
    }
    suppressBorder(edge, width, height, 3);
    removeSpecks(edge, width, height, 40); // 마스크에 남은 작은 섬의 테두리
    return dilateDisc(edge, width, height, 1.5);
  }

  // 색 경계 선: R/G/B 세 채널의 변화를 합친 색 거리로 "색이 확 바뀌는 곳"만. 능선 한 줄만 남기고(NMS)
  // 강한 선에 이어지는 약한 선만 살린다(히스테리시스). 기준은 백분위가 아니라 절대값 — 그림자처럼
  // 애매한 경계(색 거리 20~40)는 항상 빼고, 볼·무늬(150+)는 항상 잡히게. 강도 50 → 상한 85.
  function colorEdgeMask(imageData, width, height, strength) {
    const { r, g, b } = splitChannels(imageData);
    const radius = 2;
    const blur2 = (ch) => blurFractional(blurFractional(ch, width, height, radius), width, height, radius);
    const { mag, dir } = gradientColor([blur2(r), blur2(g), blur2(b)], width, height, radius);
    const ridge = nonMaxSuppress(mag, dir, width, height, radius);
    const high = 130 - strength * 0.9; // 0 → 130 (거의 안 잡힘), 100 → 40 (그림자까지)
    let mask = hysteresis(ridge, width, height, high * 0.5, high);
    suppressBorder(mask, width, height, 4); // 블러가 이미지 바깥을 가장자리 복제로 채워 생기는 가짜 선
    removeSpecks(mask, width, height, 60);
    pruneSpurs(mask, width, height, 40);
    return dilateDisc(mask, width, height, 1.5);
  }

  // 마스크(1=선)를 결과 캔버스에 그린다. 그린 뒤 지우개 자국을 다시 얹는다.
  function paintMask(mask, width, height) {
    resultCanvas.width = width;
    resultCanvas.height = height;
    const out = document.createElement('canvas');
    out.width = width;
    out.height = height;
    const octx = out.getContext('2d');
    const outData = octx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
      const v = mask[i] ? 0 : 255;
      const o = i * 4;
      outData.data[o] = v;
      outData.data[o + 1] = v;
      outData.data[o + 2] = v;
      outData.data[o + 3] = 255;
    }
    octx.putImageData(outData, 0, 0);
    resultCanvas.getContext('2d').drawImage(out, 0, 0);
    lastPaint = { mask, width, height };
    applyEraseStrokes();
    schedulePrintPrep();
  }

  // ---------- 인쇄 ----------
  // 브라우저 인쇄로 A4 한 장: 결과물 크게, 원본 우측 하단 작게, 제목 하단 중앙. PDF 저장도 인쇄 대화상자에서.
  // 함정: src를 넣자마자 print()를 부르면 data URL 디코딩이 안 끝나서 미리보기가 빈 종이로 나온다(실측).
  // 그래서 (1) 결과가 그려질 때마다 인쇄용 이미지를 미리 채워두고, (2) 버튼은 decode()를 기다린 뒤 인쇄한다.
  let printPrepTimer = null;
  function updatePrintImages() {
    if (!lastPaint) return Promise.resolve();
    printResult.src = resultCanvas.toDataURL('image/png');
    printOriginal.src = originalCanvas.width ? originalCanvas.toDataURL('image/png') : '';
    printTitle.textContent = titleInput.value.trim();
    const waits = [printResult.decode().catch(() => {})];
    if (printOriginal.getAttribute('src')) waits.push(printOriginal.decode().catch(() => {}));
    return Promise.all(waits);
  }
  function schedulePrintPrep() {
    clearTimeout(printPrepTimer);
    printPrepTimer = setTimeout(() => { updatePrintImages(); }, 400);
  }
  printBtn.addEventListener('click', async () => {
    if (!lastPaint) return;
    await updatePrintImages();
    window.print();
  });
  titleInput.addEventListener('input', () => { printTitle.textContent = titleInput.value.trim(); });
  // Ctrl+P: 미리 채워둔 이미지를 쓴다 (여기서 await는 못 하므로 최선의 노력)
  window.addEventListener('beforeprint', () => { if (!printResult.getAttribute('src')) updatePrintImages(); });

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
    const { mask, width, height } = lastPaint;
    paintMask(mask, width, height);
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

  const endStroke = () => { activeStroke = null; schedulePrintPrep(); };
  resultWrap.addEventListener('pointerup', endStroke);
  resultWrap.addEventListener('pointercancel', endStroke);
  resultWrap.addEventListener('pointerleave', () => { eraserCursor.hidden = true; });

  // ---------- 이미지 처리 헬퍼 ----------
  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // R/G/B 채널을 각각 Float32 배열로 분리
  function splitChannels(imageData) {
    const { data, width, height } = imageData;
    const n = width * height;
    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      r[p] = data[i];
      g[p] = data[i + 1];
      b[p] = data[i + 2];
    }
    return { r, g, b };
  }

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
  function blurFractional(src, width, height, radius) {
    const lo = Math.floor(radius);
    const frac = radius - lo;
    const a = boxBlur(src, width, height, lo);
    if (frac === 0) return a;
    const b = boxBlur(src, width, height, lo + 1);
    for (let i = 0; i < a.length; i++) a[i] += (b[i] - a[i]) * frac;
    return a;
  }

  // 색 변화의 세기와 방향. 세 채널의 변화 벡터를 합쳐(RGB 공간에서의 거리) 세기로 쓰고,
  // 방향은 가장 크게 변한 채널의 것을 따른다 — 방향은 NMS에서 4방향으로 양자화되므로 이 정도로 충분.
  // step은 블러 반경에 맞춘다: 블러가 경계를 폭 ~2*step의 완만한 경사로 펴놓기 때문에 이웃 1픽셀만 보면 못 잡는다.
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

  // 비최대 억제: 경사면 전체가 아니라 능선(정점)만 남긴다 — 굵은 띠가 아니라 한 줄 선이 되도록.
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

  // 히스테리시스: 강한 선(high 이상)에서 출발해 이어지는 약한 선(low 이상)만 살린다 → 흩어진 점이 아니라 이어진 선.
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

  // 가지치기: 1px 선에서 끝점을 steps번 깎고, 살아남은 끝점에서만 원래 선을 따라 steps번 되살린다.
  // → 윤곽에 매달린 길이 steps 이하 가지(주름·노이즈)는 사라지고, 긴 열린 획은 원래 길이로 복구된다.
  // 끝점에서 거슬러 걷는 방식은 계단 대각선에서 가짜 분기점을 만나서 이 방식으로 바꿨다.
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

    let frontier = [];
    for (let i = 0; i < n; i++) if (mask[i] && countNeighbors(mask, i) <= 1) frontier.push(i);
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
            if (mask[j] && countNeighbors(mask, j) <= 1 && next.indexOf(j) < 0) next.push(j);
          }
        }
      }
      frontier = next;
    }

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

  // 원형 커널 최대값 필터 (선 굵기 확장). 정사각 커널은 선이 각지게 굵어지고 크기가 껑충 뛰어서
  // (9→25→49px) 중간 굵기가 없다. 원형은 면적이 완만하게 늘어 0.5 단위 조절이 실제로 눈에 보인다.
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
