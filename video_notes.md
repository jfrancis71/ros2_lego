Some Video Notes

Image: 640x240

We use stereo split node and measure delay back to robot of /left/camera_info.

Direct raw image from cam2image to stereo split node:
cam2image uses 14% CPU.?? Have also measured 30%....
Latency for reading image on brickpi3: .01s
Network Bandwidth: 36Mb
Roundtrip latent: .08

Compressed image from cam2image to stereo split node:
cam2image uses 14% CPU.
Republish 14%
Latency for reading compressed image on brickpi3: .018s
Network Bandwidth: 4.2Mb
Roundtrip latent: .04

