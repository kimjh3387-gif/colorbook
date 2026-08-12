// 색칠공부 변환기 — 모든 처리는 브라우저 안에서만 일어난다 (업로드/네트워크 요청 없음)
(() => {
  const MAX_DIMENSION = 1600; // 성능 보호용 최대 변 길이

  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const controls = document.getElementById('controls');
  const previewEmpty = document.getElementById('previewEmpty');
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
  const sensitivityVal = document.getElementById('sensitivityVal');
  const thicknessVal = document.getElementById('thicknessVal');
  const denoiseVal = document.getElementById('denoiseVal');

  const DEFAULTS = { sensitivity: 50, thickness: 1, denoise: 1 };

  let sourceImage = null; // HTMLImageElement
  let renderQueued = false;

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
      drawOriginal();
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
    const scale = Math.min(1, MAX_DIMENSION / Math.max(w, h));
    return { width: Math.round(w * scale), height: Math.round(h * scale) };
  }

  // ---------- 컨트롤 ----------
  [
    [sensitivity, sensitivityVal],
    [thickness, thicknessVal],
    [denoise, denoiseVal],
  ].forEach(([input, out]) => {
    input.addEventListener('input', () => {
      out.textContent = input.value;
      scheduleRender();
    });
  });

  resetBtn.addEventListener('click', () => {
    sensitivity.value = DEFAULTS.sensitivity;
    thickness.value = DEFAULTS.thickness;
    denoise.value = DEFAULTS.denoise;
    sensitivityVal.textContent = DEFAULTS.sensitivity;
    thicknessVal.textContent = DEFAULTS.thickness;
    denoiseVal.textContent = DEFAULTS.denoise;
    scheduleRender();
  });

  downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'coloring-page.png';
    link.href = resultCanvas.toDataURL('image/png');
    link.click();
  });

  // ---------- 렌더 스케줄링 (슬라이더 연속 입력 debounce) ----------
  function scheduleRender() {
    if (!sourceImage) return;
    if (renderQueued) return;
    renderQueued = true;
    processingOverlay.hidden = false;
    // requestAnimationFrame에 의존하면 탭이 백그라운드/비표시 상태일 때 콜백이 멈춘다.
    // setTimeout만 사용해 화면 표시 여부와 무관하게 실행되도록 한다.
    setTimeout(() => {
      render();
      renderQueued = false;
      processingOverlay.hidden = true;
    }, 0);
  }

  // ---------- 이미지 처리 파이프라인 ----------
  function render() {
    const { width, height } = fitSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
    resultCanvas.width = width;
    resultCanvas.height = height;

    const src = document.createElement('canvas');
    src.width = width;
    src.height = height;
    const sctx = src.getContext('2d');
    sctx.drawImage(sourceImage, 0, 0, width, height);
    const imageData = sctx.getImageData(0, 0, width, height);

    let gray = toGrayscale(imageData);

    const denoiseRadius = parseInt(denoise.value, 10);
    if (denoiseRadius > 0) {
      gray = boxBlur(gray, width, height, denoiseRadius);
    }

    const magnitude = sobelMagnitude(gray, width, height);

    const sensitivityPct = parseInt(sensitivity.value, 10); // 0~100, 높을수록 선 많음
    const thresholdValue = 255 * (1 - sensitivityPct / 100) * 0.6; // 체감 곡선 보정
    let mask = thresholdMask(magnitude, thresholdValue);

    const thicknessRadius = parseInt(thickness.value, 10);
    if (thicknessRadius > 0) {
      mask = dilate(mask, width, height, thicknessRadius);
    }

    const out = document.createElement('canvas');
    out.width = width;
    out.height = height;
    const octx = out.getContext('2d');
    const outData = octx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
      const v = mask[i] ? 0 : 255; // 엣지=검정, 배경=흰색
      const o = i * 4;
      outData.data[o] = v;
      outData.data[o + 1] = v;
      outData.data[o + 2] = v;
      outData.data[o + 3] = 255;
    }
    octx.putImageData(outData, 0, 0);

    const rctx = resultCanvas.getContext('2d');
    rctx.drawImage(out, 0, 0);
  }

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

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

  function sobelMagnitude(gray, width, height) {
    const mag = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let gx = 0, gy = 0, k = 0;
        for (let ky = -1; ky <= 1; ky++) {
          const yy = clamp(y + ky, 0, height - 1);
          for (let kx = -1; kx <= 1; kx++) {
            const xx = clamp(x + kx, 0, width - 1);
            const v = gray[yy * width + xx];
            gx += v * SOBEL_X[k];
            gy += v * SOBEL_Y[k];
            k++;
          }
        }
        mag[y * width + x] = Math.sqrt(gx * gx + gy * gy);
      }
    }
    return mag;
  }

  function thresholdMask(magnitude, thresholdValue) {
    const mask = new Uint8Array(magnitude.length);
    for (let i = 0; i < magnitude.length; i++) {
      mask[i] = magnitude[i] > thresholdValue ? 1 : 0;
    }
    return mask;
  }

  // 정사각 커널 최대값 필터 (선 굵기 확장)
  function dilate(mask, width, height, radius) {
    const out = new Uint8Array(mask.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let hit = 0;
        for (let ky = -radius; ky <= radius && !hit; ky++) {
          const yy = clamp(y + ky, 0, height - 1);
          for (let kx = -radius; kx <= radius; kx++) {
            const xx = clamp(x + kx, 0, width - 1);
            if (mask[yy * width + xx]) { hit = 1; break; }
          }
        }
        out[y * width + x] = hit;
      }
    }
    return out;
  }
})();
