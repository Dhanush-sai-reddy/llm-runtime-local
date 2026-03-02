@echo off
echo Pulling WiFi DensePose Docker image...
docker pull ruvnet/wifi-densepose:latest

echo Starting WiFi DensePose container...
echo Ports: 3000 (API), 3001 (Web UI), 5005/udp (CSI data)
docker run -d --name wifi-densepose -p 3000:3000 -p 3001:3001 -p 5005:5005/udp ruvnet/wifi-densepose:latest

echo Container started! Access:
echo - Web UI: http://localhost:3001
echo - API: http://localhost:3000
echo.
echo To stop: docker stop wifi-densepose
echo To remove: docker rm wifi-densepose
pause