wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"ranknet","data":' ..
           '{"user":[[0.1,0.2,0.3]],' ..
           '"pos_item":[[0.2,0.3,0.4]],' ..
           '"neg_item":[[0.5,0.6,0.7]]}}'
