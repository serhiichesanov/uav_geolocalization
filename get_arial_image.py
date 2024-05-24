import requests

API_KEY = 'AIzaSyCifYFIfVr0QxH4aCqY6_dxgJ5B_6FEb7c'


def get_image(latitude, longitude, zoom, idx, size='640x640', scale=2):
    url = f'https://maps.googleapis.com/maps/api/staticmap?' \
          f'center={latitude},{longitude}' \
          f'&zoom={zoom}' \
          f'&size={size}' \
          f'&scale={scale}' \
          f'&maptype=satellite' \
          f'&format=png' \
          f'&key={API_KEY}'

    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f'Failed to fetch image #{idx + 1}. Status code: {response.status_code}. Content: {response.content}')
        return
