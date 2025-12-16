# Serving ML models with MLflow

Follow the steps below to launch a prediction server and visualize results on an
HTML page.

---

### 1. Start the proxy server

```bash
python proxy_server.py
```

---

### 2. Save the model artifact to MLflow

```bash
python mlflow_model.py
```

---

### 3. Launch MLflow model serving

```bash
mlflow models serve -m cats_dogs_model/ \
    --port 8888 \
    --host 0.0.0.0 \
    --no-conda
```

---

### 4. Run the HTML Page

```bash
python -m http.server 9000
```

---

## Example Request to the Running MLflow Server

```bash
curl -X POST http://localhost:8888/invocations \
     -H "Content-Type: application/json" \
     -d '{
           "inputs": [
             {
               "data": "/Users/vl.naumov/Desktop/courses/mlops_course/cats-and-dogs/mlflow-serve/uploads/e5ddb19e-d963-47f9-a443-910805ff17bf_e02e4371-092c-4f53-a90f-c60900387bdc_dog.jpg"
             }
           ]
         }'
```
