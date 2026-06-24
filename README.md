## End to End Machine Learning Project / Data science project

### Deployment

This project can be deployed using Docker.

Build the image and run the container locally:

```bash
docker build -t student-performance-app .
docker run -p 5000:5000 student-performance-app
```

Or use Docker Compose:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:5000/` in your browser.

> Make sure the Docker daemon is running before executing these commands.


## End to End Machine Learning Project / Data science project