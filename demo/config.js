const VLLM_LAB_API = ["localhost", "127.0.0.1"].includes(window.location.hostname)
  ? "http://localhost:8000"
  : "https://vllm-architecture-lab-api.onrender.com";
