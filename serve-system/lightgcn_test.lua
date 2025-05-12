wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"lightgcn","data":{"edge_index":[[0,1,2,3],[1,2,3,4]]}}'
