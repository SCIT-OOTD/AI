# 🛠️ Team Collaboration Rules

## 1. Git & Workflow Strategy

### 커밋 메시지 (Commit Convention)

<i>기본적으로 <strong>Angular JS Commit Convention</strong>을 따릅니다.</i>

#### Type:
- `Feat` : 새로운 기능 추가
- `Fix` : 버그 수정
- `Docs` : 문서 수정
- `Style` : 코드 포맷팅, 세미콜론 누락 등 (비즈니스 로직 변경 없음)
- `Refactor` : 코드 리팩토링 (기능 변경 없음)
- `Test` : 테스트 코드 추가 및 수정
- `Chore` : 빌드 설정, 패키지 매니저 설정 등
- `Design` : CSS 등 사용자 UI 디자인 변경
- `Rename` : 파일 혹은 폴더명 수정/이동

#### 규칙:
- `Type: Subject` 형식 (예: `Feat: 로그인 API 구현`)

- <strong>제목(Subject)</strong>: 50자 이내 권장, 명령문으로 작성 (예: "수정했음" -> "수정")

- <strong>본문(Body)</strong>: (선택사항) 제목에서 한 줄 띄우고 작성, '무엇을', '왜' 변경했는지 설명

### 브랜치 전략 (Branch Strategy)

<i><strong>Github Flow</strong>를 기반으로 하되, 이슈 번호를 포함하여 추적 가능하게 합니다.</i>

- <strong>Main(Master)</strong> : 배포 가능한 상태 유지
- <strong>Develop</strong> : (필요시) 다음 버전을 위한 개발 진행
- <strong>Feature</strong> : 기능 개발 브랜치
    - 명명법: `feat/issue번호-기능명` or `feat/기능명`
    - 예시: `feat/12-login`, `feat/oauth-google`
- <strong>Fix</strong> : 버그 수정 브랜치
    - 명명법: `fix/issue번호-버그명` or `fix/버그명`
    - 예시: `fix/15-typo-correction`

### Pull Request (PR) 규칙

<i>Pull Request 템플릿을 사용하여 리뷰어의 시간을 절약합니다.</i>

- <strong>Title</strong>: `[Type] 제목` (예: `[Feat] 회원가입 기능 구현`)
- <strong>내용 필수 포함 항목</strong>:
    1. <strong>작업 내용</strong> (What)
    2. <strong>관련 이슈</strong> (`Closes #이슈번호`)
    3. <strong>테스트 방법</strong> (How to test)
    4. <strong>스크린샷</strong> (UI 변경이 있는 경우)

## 2. Naming Convention (명명 규칙)

### Common
- `Class / Component` : PascalCase (예: `UserService`, `LoginButton`)
- `Variable / Method` : camelCase (예: `getUserList`, `isActive`)
- `Constant` : UPPER_SNAKE_CASE (예: `MAX_COUNT`, `API_URL`)
- `File Names` :
    - JS/TS/Java Class: PascalCase (예: `UserProfile.js`)
    - Assets/Images: snake_case (예: `main_logo.png`)

### BackEnd (Spring/Java 기준 제안)

<i>User가 작성한 `Entity: lowercase`는 Java 클래스 기준 PascalCase가 표준이므로 수정 제안합니다.</i>
- `Entity Class` : PascalCase (예: `MemberEntity`, `Order`)
- `DB Table` : snake_case (예: `member_info`, `orders`)
- `DB Column` : snake_case (예: `created_at`, `user_id`)
- `URL Endpoint` : kebab-case (예: `/api/user-profiles`)
- `DTO` : 기능+DTO (예: `UserResponseDTO`, `LoginRequestDTO`)

### FrontEnd (React 기준)
- `Component Folder` : PascalCase (예: `components/Button/index.tsx`)
- `Hook` : camelCase, `use` 접두어 필수 (예: `useAuth`)
- `Prop Types / Interface` : `I` 접두어 또는 `PascalCase` (예: `IUserProps`, `UserProps`)
- `Event Handler` : `handle` + 동사 (예: `handleSubmit`, `handleClick`)

## 3. API Response Standard (협업 필수)

<i>프론트엔드와 백엔드가 통신할 때 혼란을 줄이기 위해 응답 형식을 통일합니다.</i>

모든 API 응답은 아래 JSON 구조를 기본으로 합니다.

```json
{
  "status": 200,              // HTTP Status Code
  "message": "성공하였습니다.", // 응답 메시지 (에러 시 에러 사유)
  "data": {                   // 실제 데이터 (없으면 null)
    "id": 1,
    "name": "User"
  }
}
```

<strong>성공 시:</strong> `200 OK`와 함께 요청 데이터 반환

<strong>실패 시:</strong> `4xx`, `5xx`와 함께 `data`는 `null`, `message`에 에러 내용 포함

## 4. Code Quality & Formatting

<i>사람이 검사하지 않고 도구가 검사하게 합니다.</i>
- <strong>Formatter:</strong>
    - FE: <strong>Prettier</strong> 사용 (탭 간격 2, 세미콜론 사용, 작은따옴표 사용 등 `.prettierrc` 공유)
    - BE: IDE 기본 Formatter 혹은 <strong>Google Java Style Guide</strong> 적용
- <strong>Linter:</strong>
    - FE: <strong>ESLint</strong> (사용하지 않는 변수 금지, `console.log` 금지 등)
    - BE: <strong>SonarLint</strong> (권장)
- <strong>Comment (주석):</strong>
    - 복잡한 로직이나 <strong>'왜(Why)'</strong> 이렇게 짰는지에 대한 설명이 필요할 때만 작성합니다.
    - 뻔한 코드(Get/Set)에는 주석을 달지 않습니다.

## 5. Security & Environment
- <strong>Secrets:</strong> API Key, DB Password 등 민감 정보는 절대 Git에 올리지 않습니다.
    - `.env` 파일을 활용하고 `.gitignore`에 등록합니다.
    - 팀원 간 공유는 메신저나 보안 채널(`Notion` 등)을 이용합니다.