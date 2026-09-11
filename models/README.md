# models/

- `pidinet_tiny.onnx` (0.4MB) — PiDiNet tiny (ICCV 2021, "Pixel Difference Networks for Efficient Edge Detection", BSDS500 학습).
  - 출처: https://huggingface.co/bdck/PiDiNet_ONNX 의 `table5_pidinet_tiny.onnx` + `.onnx.data` 를
    `onnx.save_model(..., save_as_external_data=False)` 로 파일 하나로 합친 것 (onnxruntime-web이 외부 데이터 없이 바로 읽게).
  - 원 저장소 https://github.com/hellozhuo/pidinet (MIT).
  - 입력 `image` [1,3,H,W] float32, ImageNet mean/std 정규화. 출력 `fused` [1,1,H,W] = 픽셀별 윤곽 확률 0~1.
- base 모델(`pidinet_table5`, 3MB)도 써봤는데 WebGPU에서 5배 느리고(1.3초 vs 0.25초) 선 품질은 거의 같아서 tiny만 넣었다.
