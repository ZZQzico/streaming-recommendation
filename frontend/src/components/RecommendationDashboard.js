import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography, Slider, Button, Card, CardContent, CardMedia, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { recommendationService } from '../services/api';

// Generate mock recommendations based on timestamp
const generateMockRecommendations = (timestamp) => {
  const books = [
    { title: 'The Great Adventure', category: 'Fiction' },
    { title: 'Science Today', category: 'Science' },
    { title: 'History of Time', category: 'Non-Fiction' },
    { title: 'Digital Age', category: 'Technology' },
    { title: 'Modern Philosophy', category: 'Philosophy' },
    { title: 'Art of Cooking', category: 'Cooking' }
  ];

  // Use timestamp to vary the recommendations
  const startIndex = timestamp % (books.length - 3);
  return books.slice(startIndex, startIndex + 3).map((book, index) => ({
    id: timestamp + index,
    title: book.title,
    rating: (3 + Math.sin(timestamp / 10 + index) * 1.5).toFixed(1),
    category: book.category
  }));
};

const RecommendationDashboard = () => {
  const [timeSlider, setTimeSlider] = useState(0);
  const [recommendations, setRecommendations] = useState(generateMockRecommendations(0));
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [analyticsData, setAnalyticsData] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);

  // Fetch recommendations based on time
  const fetchRecommendations = async (timestamp) => {
    try {
      const data = await recommendationService.getRecommendations(timestamp);
      setRecommendations(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
      // Use mock data when API fails
      setRecommendations(generateMockRecommendations(timestamp));
      setError('Using simulated recommendations (backend not connected)');
    }
  };

  // Handle time slider change
  const handleTimeSliderChange = async (event, newValue) => {
    setTimeSlider(newValue);
    await fetchRecommendations(newValue);
  };

  // Handle speed change
  const handleSpeedChange = async (speed) => {
    setReplaySpeed(speed);
    try {
      await recommendationService.setReplaySpeed(speed);
      setError(null);
    } catch (err) {
      console.error('Failed to set replay speed:', err);
      setError('Failed to set replay speed, but UI is updated.');
    }
  };

  // Toggle play/pause
  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  // Auto-advance time when playing
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(async () => {
        setTimeSlider((prev) => {
          const next = prev + replaySpeed;
          if (next >= 100) {
            setIsPlaying(false);
            return 100;
          }
          return next;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, replaySpeed]);

  // Update recommendations when time changes
  useEffect(() => {
    fetchRecommendations(timeSlider);
  }, [timeSlider]);

  // Generate and update analytics data
  useEffect(() => {
    const generateAnalytics = () => {
      const data = [];
      for (let i = 0; i <= timeSlider; i += 5) {
        data.push({
          time: `${Math.floor(i / 60)}:${(i % 60).toString().padStart(2, '0')}`,
          score: (3 + Math.sin(i / 10) * 1.5).toFixed(1)
        });
      }
      return data;
    };

    try {
      recommendationService.getAnalytics(0, timeSlider)
        .then(data => setAnalyticsData(data))
        .catch(() => {
          setAnalyticsData(generateAnalytics());
        });
    } catch {
      setAnalyticsData(generateAnalytics());
    }
  }, [timeSlider]);

  return (
    <Grid container spacing={3}>
      {/* Time Control Section */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Time Control
          </Typography>
          {error && (
            <Typography color="error" variant="body2" gutterBottom>
              {error}
            </Typography>
          )}
          <Slider
            value={timeSlider}
            onChange={handleTimeSliderChange}
            min={0}
            max={100}
            valueLabelDisplay="auto"
          />
          <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
            <Button 
              variant="contained" 
              onClick={togglePlay}
              color={isPlaying ? "secondary" : "primary"}
            >
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button 
              variant={replaySpeed === 1 ? "contained" : "outlined"}
              onClick={() => handleSpeedChange(1)}
            >
              1x
            </Button>
            <Button 
              variant={replaySpeed === 2 ? "contained" : "outlined"}
              onClick={() => handleSpeedChange(2)}
            >
              2x
            </Button>
            <Button 
              variant={replaySpeed === 5 ? "contained" : "outlined"}
              onClick={() => handleSpeedChange(5)}
            >
              5x
            </Button>
            <Typography variant="body2">
              Time Point: {timeSlider}
            </Typography>
          </Box>
        </Paper>
      </Grid>

      {/* Recommendations Section */}
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Recommendations (Time Point: {timeSlider})
          </Typography>
          <Grid container spacing={2}>
            {recommendations.map((item) => (
              <Grid item xs={12} sm={6} md={4} key={item.id}>
                <Card>
                  <CardMedia
                    component="img"
                    height="140"
                    image={`https://picsum.photos/seed/${item.id}/200/300`}
                    alt={item.title}
                  />
                  <CardContent>
                    <Typography gutterBottom variant="h6" component="div">
                      {item.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Rating: {item.rating}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Category: {item.category}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Analytics Section */}
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Analytics
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analyticsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="score" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default RecommendationDashboard; 