import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://plant-disease-detection-cwoo.onrender.com/api';

const api = axios.create({ baseURL: API_BASE_URL });

export const predictImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/predict', formData);
};

export const getHistory = (limit = 50) => api.get(`/history?limit=${limit}`);
export const getStats = () => api.get('/stats');
export const getModelInfo = () => api.get('/model-info');
export const getDiseaseInfo = () => api.get('/disease-info');
export const getHealth = () => api.get('/health');
export const getResultUrl = (name) => `${API_BASE_URL}/results/${encodeURIComponent(name)}`;

export default api;
