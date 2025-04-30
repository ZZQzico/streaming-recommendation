import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Box } from '@mui/material';
import { Refresh as RefreshIcon, Settings as SettingsIcon } from '@mui/icons-material';

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Streaming Recommendations
        </Typography>
        <Box>
          <IconButton color="inherit" aria-label="refresh">
            <RefreshIcon />
          </IconButton>
          <IconButton color="inherit" aria-label="settings">
            <SettingsIcon />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 