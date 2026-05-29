# Trueframe System Architecture

Trueframe is an authenticity-first social media ecosystem designed to eradicate deepfakes and misinformation at the source. This document provides a multi-perspective architectural breakdown of the platform.

---

## 1. High-Level System Overview
This diagram illustrates the primary communication paths between the user, the application layers, and the cloud infrastructure.

```mermaid
graph LR
    User((User)) -- Interacts --> FE[Frontend: React SPA]
    FE -- Auth/SSO --> AU[Supabase Auth]
    FE -- REST API --> BE[Backend: Fastify Node.js]
    
    subgraph "Orchestration Layer"
        BE -- Spawn --> AI[AI Service: Python]
        BE -- Query --> DB[(Supabase DB: PostgreSQL)]
        BE -- Upload --> ST[(Supabase Storage: S3)]
    end
    
    AI -- Weights --> HF[HuggingFace Hub]
```

### Component Summary:
- **Frontend**: Built with Vite, React, and Tailwind. Handles real-time verification progress and trust-ranked feed rendering.
- **Backend**: An orchestration layer using Fastify. Manages the "fail-closed" verification logic and data integrity.
- **AI Service**: A modular Python engine performing deep feature extraction and forgery detection.
- **Supabase**: Unified backend providing PostgreSQL, S3 Storage, and JWT-based Authentication.

---

## 2. Deep AI Verification Pipeline
This diagram provides a deep-dive into the multi-stage detection process that occurs for every piece of content uploaded to Trueframe.

```mermaid
graph TD
    Upload([New Media Uploaded]) --> M_Pre[Metadata Scanner]
    
    M_Pre -->|Integrity Check| F_Ext[Frame Extractor]
    F_Ext -->|Max 32 Frames @ 1FPS| F_Anl[Face Analyzer]
    
    subgraph "Detection Core (Parallel Analysis)"
        F_Anl --> S1[EfficientNet ONNX - Spatial Forgery]
        F_Anl --> S2[Azimuthal FFT - Frequency Artifacts]
        F_Anl --> S3[Noise Residual Analysis - Local Patches]
    end
    
    S1 & S2 & S3 --> Fusion{Weighted Score Fusion}
    
    Fusion -->|Success| Temp[Temporal Consistency Check]
    Fusion -- Rejected --> Rej([Content Blocked])
    
    Temp --> Trust[Trust Score Orchestrator]
    Trust -->|Update Score| DB_Write[(PostgreSQL: verifications)]
    DB_Write --> Publish([Live on Feed])
```

### Deep Analysis Details:
- **Spatial Forgery**: Uses EfficientNet-B0 to detect pixel-level inconsistencies, blending borders, and unnatural textures.
- **Frequency Artifacts**: Analyzes the Power Spectral Density (PSD) using Azimuthal Integration to find periodic noise patterns typical of GAN-generated content.
- **Temporal Consistency**: (For Video) Ensures that the verification verdict remains stable across multiple frames to prevent "stitching" attacks.
- **Fail-Closed Logic**: If any component (AI model, timeout, or extractor) fails, the content is automatically rejected by default.

---

## 3. Data Flow & Security
This diagram focuses on the movement of sensitive data and the security boundaries.

```mermaid
sequenceDiagram
    participant U as User Browser
    participant S as Supabase Auth
    participant B as Fastify Backend
    participant A as AI Service
    participant D as PostgreSQL

    U->>S: Authenticate (Login/SSO)
    S-->>U: JWT Token
    U->>B: POST /upload (JWT + Media)
    B->>B: Verify JWT Signature
    B->>A: spawn subprocess(media_path)
    A->>A: Multi-stage Detection
    A-->>B: JSON Result (Verdict + Confidence)
    B->>D: Update Authenticity Score
    D-->>B: RLS Check (Owner Only)
    B-->>U: 200 OK (Status: VERIFIED)
```

### Security Highlights:
- **JWT Enforcement**: All API routes are protected by JWT verification against Supabase Auth.
- **Row-Level Security (RLS)**: PostgreSQL tables are locked down so users can only modify their own verification records.
- **Isolated AI Process**: The AI engine runs as a separate subprocess with restricted permissions, preventing potential remote code execution via malicious media files.

---

## 4. Technology Stack
| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS, Shadcn UI, TanStack Query |
| **Backend** | Node.js, Fastify, TSX, Zod (Validation), Redis (Caching) |
| **AI / ML** | Python 3.13, PyTorch, ONNX Runtime, OpenCV, MediaPipe |
| **Infrastructure** | Supabase (PostgreSQL, Storage, Auth), Vercel |

---

## 5. Visual Excalidraw Diagram
For a fully interactive visual representation, refer to the generated `.excalidraw` file in the project workspace:
- `detailed-architecture.excalidraw`
