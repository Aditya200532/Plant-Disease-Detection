import axios from 'axios';
const api = axios.create({
  baseURL: 'https://plant-disease-detection-cwoo.onrender.com/api',
});

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

export default api;
