// 색칠공부 변환기 — 모든 처리는 브라우저 안에서만 일어난다 (업로드/네트워크 요청 없음)
(() => {
  const MAX_DIMENSION = 1600; // 성능 보호용 최대 변 길이

  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
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
  const adaptive = document.getElementById('adaptive');
  const shading = document.getElementById('shading');
  const sensitivityVal = document.getElementById('sensitivityVal');
  const thicknessVal = document.getElementById('thicknessVal');
  const denoiseVal = document.getElementById('denoiseVal');
  const adaptiveVal = document.getElementById('adaptiveVal');

  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxClose = document.getElementById('lightboxClose');

  // denoise 기본값 3: 6이면 해·광배처럼 가늘고 대비 약한 요소가 흐림 단계에서 통째로 지워진다
  // (측정: 단순화 2~4에서는 해가 잡히고 6에서는 0). 3이면 텍스처는 눌리면서 해는 살아남는다.
  const DEFAULTS = { sensitivity: 50, thickness: 1, denoise: 3, adaptive: 60, shading: false };

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
    [adaptive, adaptiveVal],
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
    adaptive.value = DEFAULTS.adaptive;
    shading.checked = DEFAULTS.shading;
    sensitivityVal.textContent = DEFAULTS.sensitivity;
    thicknessVal.textContent = DEFAULTS.thickness;
    denoiseVal.textContent = DEFAULTS.denoise;
    adaptiveVal.textContent = DEFAULTS.adaptive;
    scheduleRender();
  });

  // ---------- 크게 보기(돋보기) ----------
  document.querySelectorAll('.zoom-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
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

  // ---------- 렌더 스케줄링 (슬라이더 연속 입력 debounce) ----------
  function scheduleRender() {
    if (!sourceImage) return;
    if (renderQueued) return;
    renderQueued = true;
    processingOverlay.textContent = '변환 중…';
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
    const { width, height } = fitSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
    resultCanvas.width = width;
    resultCanvas.height = height;

    const src = document.createElement('canvas');
    src.width = width;
    src.height = height;
    const sctx = src.getContext('2d', { willReadFrequently: true });
    sctx.drawImage(sourceImage, 0, 0, width, height);
    const imageData = sctx.getImageData(0, 0, width, height);

    const gray = toGrayscale(imageData);

    // 1) 단순화: 반경이 클수록 잔 디테일이 사라져 아이들용 단순한 그림이 된다.
    //    박스 블러 2회는 가우시안 블러의 값싼 근사. 반경은 0.5 단위(소수)까지 받는다.
    const radius = Math.max(0.5, parseFloat(denoise.value));
    const smooth = blurFractional(blurFractional(gray, width, height, radius), width, height, radius);

    // 2) 블러 반경만큼 떨어진 픽셀끼리 비교한다. 블러가 경계를 폭 ~2*radius의 완만한
    //    경사로 펴놓기 때문에, 이웃 1픽셀만 보는 고정 3x3 커널로는 그 경사를 못 잡는다.
    const step = Math.max(1, Math.round(radius));
    const { mag, dir } = gradient(smooth, width, height, step);

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

    // 7) 굵기
    const thicknessRadius = parseFloat(thickness.value);
    if (thicknessRadius > 0) {
      mask = dilateDisc(mask, width, height, thicknessRadius);
    }

    // 음영(옵션): 어두운 영역에 연한 회색 플랫톤만 깐다. 해칭 노이즈가 아니라 면 단위 채우기.
    let shadeMask = null;
    if (shading.checked) {
      let sum = 0;
      for (let i = 0; i < smooth.length; i++) sum += smooth[i];
      const shadeThreshold = (sum / smooth.length) * 0.65;
      shadeMask = new Uint8Array(smooth.length);
      for (let i = 0; i < smooth.length; i++) {
        shadeMask[i] = smooth[i] < shadeThreshold ? 1 : 0;
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
