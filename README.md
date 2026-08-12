# 색칠공부 변환기 (Coloring Page Maker)

사진을 올리면 색을 다 빼고 **색칠공부(coloring book) 윤곽선 그림**으로 바꿔주는 정적 웹 도구.
서버가 없다 — 업로드한 이미지는 브라우저 안(Canvas)에서만 처리되고 어디로도 전송되지 않는다.

## 동작 방식
1. 이미지를 grayscale로 변환
2. 잡티 제거용 박스 블러 (denoise 슬라이더)
3. Sobel 엣지 검출로 윤곽선 추출
4. 민감도 슬라이더 기준으로 이진화(threshold) → 선/배경만 남김
5. 선 굵기 슬라이더만큼 확장(dilate)
6. 흰 배경 + 검은 선으로 렌더링, PNG 다운로드

핵심 로직은 [script.js](script.js) 하나에 다 있음 (외부 라이브러리 없음, 순수 JS + Canvas API).

## 로컬에서 열어보기
그냥 `index.html`을 더블클릭해서 브라우저로 열면 된다. 별도 서버/빌드 필요 없음.

## 배포 (GitHub Pages)
- 목표 레포: `https://github.com/kimjh3387-gif/colorbook.git`
- 이 폴더 전체를 그 레포에 push → 레포 Settings → Pages → main 브랜치 루트(`/`) 로 설정하면
  `https://kimjh3387-gif.github.io/colorbook/` 에서 바로 접근 가능.
- 계정 로그인이 필요한 단계(레포 생성 확인, push 인증, Pages 설정)는 사용자가 직접 진행.

## 향후 아이디어 (용돈벌이용)
- 애드센스 등 광고 붙이기 (레이아웃에 광고 슬롯 자리 고려 필요)
- 결과물 워터마크 옵션 (무료/유료 구분 시)
- 변환 스타일 추가: 흑백 명암(그레이스케일) 버전, 포스터화 버전 등
