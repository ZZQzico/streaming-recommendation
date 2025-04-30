# Streaming Recommendation Frontend

This is the frontend application for the Streaming Recommendation System. It provides a modern, responsive interface for viewing and interacting with real-time book recommendations.

## Features

- Real-time recommendation display
- Time-based replay control
- Speed adjustment for replay
- Analytics visualization
- Responsive design for all screen sizes

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn package manager

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/streaming-recommendation.git
cd streaming-recommendation/frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Create a `.env` file in the frontend directory:
```bash
REACT_APP_API_BASE_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm start
# or
yarn start
```

The application will be available at `http://localhost:3000`.

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── Header.js
│   │   └── RecommendationDashboard.js
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── index.js
│   └── reportWebVitals.js
├── .env
└── package.json
```

## Available Scripts

- `npm start`: Runs the app in development mode
- `npm test`: Launches the test runner
- `npm run build`: Builds the app for production
- `npm run eject`: Ejects from create-react-app

## Environment Variables

- `REACT_APP_API_BASE_URL`: Backend API base URL

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 