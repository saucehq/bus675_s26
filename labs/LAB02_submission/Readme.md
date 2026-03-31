# Lab 2 Submission README

## Student Information
- Name: Alexxis Saucedo
- Date: 2026-03-31

## Deliverables Included
- `inference_api/Dockerfile`
- `preprocessor/Dockerfile`
- `inference_api/app.py` (with `/health` and `/stats`)
- `sample_classifications_20.jsonl` (first 20 lines from logs)
- `Reflection.md`

## Docker Build Commands Used

### Inference API
```bash
docker build -t congo-inference-api ./inference_api
```

### Preprocessor
```bash
docker build -t congo-preprocessor ./preprocessor
```

## Docker Run Commands Used

### Inference API Container
```bash
docker run -d --name inference-api -p 8000:8000 -v C:\Users\Lexi\logs:/logs congo-inference-api
```

### Preprocessor Container
```bash
docker run -d --name preprocessor -v C:\Users\Lexi\incoming:/incoming -e API_URL=http://host.docker.internal:8000 -e PYTHONUNBUFFERED=1 congo-preprocessor
```

## Brief Explanation: How the Containers Communicate
The preprocessor container watches the incoming folder for new images and send them to the inference API using HTTP POST requests to the predict endpoint. I pass in the API_URL environment variable when I run the container. For data persistence, both containers use binds mounts so the incoming and logs folders live on the host and don't disappear when the containers stop or restart. 

Points to cover:
- Which container calls which endpoint.
- How the preprocessor knows where to find the inference API.
- How images and logs persist using mounted host folders.
- Why `localhost` can be tricky inside containers.

