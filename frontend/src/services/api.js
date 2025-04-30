import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const recommendationService = {
  getRecommendations: async (timestamp) => {
    try {
      const response = await api.get('/recommendations', {
        params: { timestamp },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      throw error;
    }
  },

  setReplaySpeed: async (speed) => {
    try {
      const response = await api.post('/replay/speed', { speed });
      return response.data;
    } catch (error) {
      console.error('Error setting replay speed:', error);
      throw error;
    }
  },

  getUserProfile: async (userId) => {
    try {
      const response = await api.get(`/users/${userId}/profile`);
      return response.data;
    } catch (error) {
      console.error('Error fetching user profile:', error);
      throw error;
    }
  },

  getAnalytics: async (startTime, endTime) => {
    try {
      const response = await api.get('/analytics', {
        params: { startTime, endTime },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching analytics:', error);
      throw error;
    }
  },
};

export default api; 