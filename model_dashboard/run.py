from app import app

if __name__ == '__main__':
    print("Starting Recommendation System Model Evaluation Dashboard...")
    print("Access URL: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080) 