Some Video Notes

Image: 640x240

We use stereo split node and measure delay back to robot of /left/camera_info.

### Direct raw image from cam2image to stereo split node:

cam2image uses 14% CPU.?? Have also measured 30%....

Latency for reading image on brickpi3: .01s

Network Bandwidth: 36Mb

Roundtrip latent: .08


### Compressed image from cam2image to stereo split node:

cam2image uses 14% CPU.

Republish 14%

Latency for reading compressed image on brickpi3: .018s

Network Bandwidth: 4.2Mb
Roundtrip latent: .04

USB Bandwidth: 140KB (equiv 1.1Mb)

### USBCAM:

25% CPU.

Network Bandwidth: 3.0Mb

USB Bandwidth: 140KB (equiv 1.1Mb)

Roundtrip latent: .88


Observations: Anecdotally, just looking it looks to me like USBCAM has much lower latency than cam2image. Reason for discrepancy?
The USB numbers look questionable, there is some suggestion on GitHub questioning reliability.
