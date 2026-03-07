# TrueFrame - Authenticity-First Social Media Platform

TrueFrame is a next-generation social media platform built around authenticity. It uses AI deepfake detection to ensure that all media uploaded to the platform is verified as real, combating misinformation and AI-generated fakes.

## 🚀 Key Features

*   **AI Fake Detection Engine:** Automatically scans every image and video upload using advanced models (e.g., EfficientNet, CLIP) to detect deepfakes, manipulations, and AI-generated content.
*   **Trust Scores & Authenticity Labels:** Users build a global "Trust Score" based on their upload history. Content is tagged with clear Authenticity Labels (e.g., `VERIFIED_REAL`, `AI_GENERATED`).
*   **Community Fact-Checking:** A decentralized verification system where trusted users can flag and review potentially misleading or out-of-context posts.
*   **Deepfake Alerts:** Global alerts notify users about trending deepfakes or misinformation campaigns.
*   **Consolidated Schema:** Powered by a unified Supabase PostgreSQL schema with advanced social features like follows, comments, shares, and content proofs.

## 🛠️ Technologies Used

### Frontend
*   React + Vite
*   TypeScript
*   Tailwind CSS
*   shadcn-ui

### Backend & AI
*   Node.js & Express (API Layer)
*   Python (FastAPI AI microservice)
*   Supabase (PostgreSQL Database & Authentication)

## 📦 Getting Started

### Prerequisites
*   Node.js (v18+)
*   Python (3.9+)
*   Supabase CLI (optional, for local database)

### Installation

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/your-username/trueframe.git
    cd trueframe/verified-stream
    ```

2.  **Install Frontend & API Dependencies**
    ```sh
    npm install
    cd backend && npm install
    ```

3.  **Setup the AI Service**
    ```sh
    cd ai_service
    pip install -r requirements.txt
    ```

4.  **Database Setup**
    *   Create a Supabase project.
    *   Run the unified schema located at `backend/db/schema.sql` in your Supabase SQL editor.
    *   Update `.env` files with your Supabase keys.

5.  **Run Development Servers**
    *   Start Frontend: `npm run dev` (in the root directory)
    *   Start Backend: `npm run dev` (in the `backend` directory)
    *   Start AI Service: `python main.py` (in the `ai_service` directory)

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📄 License
This project is licensed under the MIT License.
