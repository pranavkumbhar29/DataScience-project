import sys
try:
    import requests
except Exception:
    print('requests_missing')
    sys.exit(2)

url='http://127.0.0.1:5000/predictdata'
data={
 'gender':'female',
 'race_ethnicity':'group B',
 'parental_level_of_education':"bachelor's degree",
 'lunch':'standard',
 'test_preparation_course':'none',
 'writing_score':'74',
 'reading_score':'72'
}
try:
    resp=requests.post(url,data=data,timeout=10)
    print('STATUS',resp.status_code)
    text=resp.text
    if 'THE prediction is' in text:
        start=text.find('THE prediction is')
        print(text[start:start+200])
    elif 'prediction is' in text:
        start=text.find('prediction is')
        print(text[start:start+200])
    else:
        print('\n...TAIL...\n')
        print(resp.text[-800:])
except Exception as e:
    print('ERROR',e)
    sys.exit(3)
