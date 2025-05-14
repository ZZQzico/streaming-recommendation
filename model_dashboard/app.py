from flask import Flask, render_template, jsonify
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型评估模块
from model_dashboard.model_evaluation import (
    evaluate_lightgcn,
    evaluate_din,
    evaluate_ranknet
)

app = Flask(__name__)

@app.route('/')
def index():
    """首页 - 显示Dashboard"""
    return render_template('dashboard.html')

@app.route('/api/metrics/lightgcn')
def lightgcn_metrics():
    """获取LightGCN模型评估指标"""
    metrics = evaluate_lightgcn()
    return jsonify(metrics)

@app.route('/api/metrics/din')
def din_metrics():
    """获取DIN模型评估指标"""
    metrics = evaluate_din()
    return jsonify(metrics)

@app.route('/api/metrics/ranknet')
def ranknet_metrics():
    """获取RankNet模型评估指标"""
    metrics = evaluate_ranknet()
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 