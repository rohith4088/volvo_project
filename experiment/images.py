from simple_image_download import simple_image_download as smp

response = smp.simple_image_download
keywords = ['volvo construction bulldozer']
for kw in keywords:
    response().download(kw,200)


