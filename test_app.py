from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_api_info_loads_artifact():
    response = client.get('/api/info')
    assert response.status_code == 200
    data = response.json()
    assert data['ready'] is True
    assert data['threshold'] == 0.59
    assert 'features' in data and len(data['features']) == 3
    assert 'importances' in data


def test_predict_valid_input():
    payload = {'absolute_magnitude': 17.5, 'relative_velocity': 90000, 'miss_distance': 5000000}
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['label'] in {'Hazardous', 'Non-hazardous'}
    assert 0.0 <= data['score'] <= 1.0
    assert data['diameter_m'] > 0
    assert data['miss_ld'] > 0


def test_predict_rejects_out_of_range_inputs():
    response = client.post('/api/predict', json={'absolute_magnitude': 2, 'relative_velocity': 1, 'miss_distance': 1})
    assert response.status_code == 422


def test_sensitivity_endpoint_returns_curves():
    payload = {'absolute_magnitude': 20, 'relative_velocity': 60000, 'miss_distance': 20000000, 'feature': 'absolute_magnitude'}
    response = client.post('/api/sensitivity', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data['x']) == len(data['y'])
    assert data['threshold'] == 0.59


def test_template_download_is_csv():
    response = client.get('/api/template')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/csv')
    body = response.text
    assert 'absolute_magnitude' in body
    assert 'relative_velocity' in body
    assert 'miss_distance' in body


def test_batch_classification_accepts_valid_csv():
    csv = b'absolute_magnitude,relative_velocity,miss_distance\n17.5,90000,5000000\n18,60000,60000000\n26,40000,50000000\n'
    response = client.post('/api/batch', files={'file': ('valid.csv', csv, 'text/csv')})
    assert response.status_code == 200
    data = response.json()
    assert data['summary']['total'] == 3
    assert set(data['summary']).issuperset({'total', 'hazardous', 'non_hazardous', 'invalid'})


def test_batch_classification_rejects_empty_file():
    response = client.post('/api/batch', files={'file': ('empty.csv', b'', 'text/csv')})
    assert response.status_code == 400
    assert 'empty' in response.json()['detail'].lower()


def test_batch_classification_rejects_missing_columns():
    csv = b'a,b\n1,2\n'
    response = client.post('/api/batch', files={'file': ('missing.csv', csv, 'text/csv')})
    assert response.status_code == 400
    assert 'Missing columns' in response.json()['detail']


def test_root_page_serves_frontend():
    response = client.get('/')
    assert response.status_code == 200
    assert '<html' in response.text.lower()
