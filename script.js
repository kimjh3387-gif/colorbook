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
  const shading = document.getElementById('shading');
  const sensitivityVal = document.getElementById('sensitivityVal');
  const thicknessVal = document.getElementById('thicknessVal');
  const denoiseVal = document.getElementById('denoiseVal');

  const DEFAULTS = { sensitivity: 50, thickness: 1, denoise: 6, shading: false };

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

  shading.addEventListener('change', scheduleRender);

  resetBtn.addEventListener('click', () => {
    sensitivity.value = DEFAULTS.sensitivity;
    thickness.value = DEFAULTS.thickness;
    denoise.value = DEFAULTS.denoise;
    shading.checked = DEFAULTS.shading;
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
      try {
        render();
        processingOverlay.hidden = true;
      } catch (err) {
        // 에러가 나도 "변환 중"에 무한히 멈춰있지 않고 사용자에게 보여준다.
        console.error('색칠공부 변환 실패:', err);
        processingOverlay.hidden = false;
        processingOverlay.textContent = '변환 중 오류가 났어요: ' + err.message;
      } finally {
        renderQueued = false;
      }
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

    const rawGray = toGrayscale(imageData);

    const denoiseRadius = parseInt(denoise.value, 10);
    // sigma1(약한 블러)은 denoise 슬라이더 그대로, sigma2(강한 블러)는 항상 그 2배.
    // 두 블러의 "차이(Difference of Gaussians)"를 윤곽선으로 쓴다.
    //  - 매끈한 그라데이션(하늘, 조명 등)은 어떤 블러를 먹여도 그대로라 차이가 0에 가까움
    //  - 잔 텍스처는 sigma1 단계에서 이미 뭉개지므로 두 블러의 차이가 거의 없음
    //  - 실제 큰 덩어리 경계만 두 블러의 폭 차이만큼 뚜렷한 차이로 남는다
    const sigma1 = Math.max(1, denoiseRadius);
    const sigma2 = sigma1 * 2;
    const grayFine = boxBlur(rawGray, width, height, sigma1);
    const grayCoarse = boxBlur(grayFine, width, height, sigma2 - sigma1);
    let gray = grayFine; // 음영 등 이후 단계에서 쓸 "기본 명암"

    const magnitude = new Float32Array(width * height);
    for (let i = 0; i < magnitude.length; i++) {
      magnitude[i] = Math.abs(grayFine[i] - grayCoarse[i]);
    }

    const sensitivityPct = parseInt(sensitivity.value, 10); // 0~100, 높을수록 선 많음
    const thresholdValue = 12 * (1 - sensitivityPct / 100) + 2; // DoG 응답 스케일에 맞춘 값
    let mask = thresholdMask(magnitude, thresholdValue);
    // 박스 블러는 이미지 바깥을 가장자리 픽셀 복제로 채우는데, sigma1/sigma2가 서로 다른
    // 만큼 그 복제 방식이 어긋나면서 실제 경계가 없어도 테두리를 따라 가짜 선이 생긴다.
    // sigma2 폭만큼 테두리를 판단에서 제외해 액자 효과를 없앤다.
    suppressBorder(mask, width, height, sigma2);

    const thicknessRadius = parseInt(thickness.value, 10);
    if (thicknessRadius > 0) {
      mask = dilate(mask, width, height, thicknessRadius);
    }

    // 음영: 켜져 있으면 어두운 영역(블러된 명암 기준, 평균보다 확 어두운 곳)에
    // 옅은 회색 플랫톤만 깔아준다. 노이즈성 해칭이 아니라 면 단위 채우기.
    let shadeMask = null;
    if (shading.checked) {
      let sum = 0;
      for (let i = 0; i < gray.length; i++) sum += gray[i];
      const mean = sum / gray.length;
      const shadeThreshold = mean * 0.65;
      shadeMask = new Uint8Array(gray.length);
      for (let i = 0; i < gray.length; i++) {
        shadeMask[i] = gray[i] < shadeThreshold ? 1 : 0;
      }
    }

    const SHADE_TONE = 225;

    const out = document.createElement('canvas');
    out.width = width;
    out.height = height;
    const octx = out.getContext('2d');
    const outData = octx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
      let v;
      if (mask[i]) {
        v = 0; // 엣지 = 검정
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

  function suppressBorder(magnitude, width, height, margin) {
    const m = Math.min(margin, Math.floor(Math.min(width, height) / 2));
    if (m <= 0) return;
    for (let y = 0; y < height; y++) {
      const inBand = y < m || y >= height - m;
      const row = y * width;
      for (let x = 0; x < width; x++) {
        if (inBand || x < m || x >= width - m) magnitude[row + x] = 0;
      }
    }
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
