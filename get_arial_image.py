import requests

API_KEY = 'AIzaSyCBhhDaxVzYRfNycGVpNEZRS2_td_JpW50'


def get_image(latitude, longitude, zoom, idx, size='512x512'):
    url = f'https://maps.googleapis.com/maps/api/staticmap?' \
          f'center={latitude},{longitude}' \
          f'&zoom={zoom}' \
          f'&size={size}' \
          f'&maptype=satellite' \
          f'&key={API_KEY}'

    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f'Failed to fetch image #{idx + 1}. Status code: {response.status_code}. Content: {response.content}')
        return
