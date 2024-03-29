import requests
import time
from bs4 import BeautifulSoup

page = 1
while page < 2:
    fout = open('huanqiuPage'+str(page)+'.txt', 'w', encoding='utf-8')
    query = "武汉 疫情 site:huanqiu.com"
    header = {'user-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}
    url = 'https://google.com/search'
    param = {'q': query,
            'newwindow': '1',
            'tbs': 'cdr:1,cd_min:1/23/2020,cd_max:3/25/2020',
            'sxsrf': 'ALeKk01MXG0m_c7Ss9x_kcOdV7hoMXepvw:1587324847979',
            'ei': 'r6ecXp6hO83SsAfA7KnABQ',
            'start': str((page-1)*10),
            'sa': 'N',
            'ved': '2ahUKEwjetNPtnfXoAhVNKewKHUB2Clg4ChDy0wN6BAgLEDM',
            'biw': '855',
            'bih': '937'
            }
    req = requests.get(url, headers=header, params=param)
    print(page)
    if req.status_code == 200:
        soup = BeautifulSoup(req.content, 'lxml')
        # print(req.content)
        for g in soup.find_all('div', class_='r'):
            # print(g)
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text
                print(link, title)
                try:
                    req = requests.get(url=link, headers=header)
                    req.encoding = 'utf-8'
                    soup = BeautifulSoup(req.text, 'lxml')
                    content = soup.find('div', class_='l-con clear')
                    pars = content.find_all('p')

                    fout.write(title)
                    fout.write("\n")
                    fout.write("\n".join([str(par) for par in pars]))
                    fout.write("\n*\n")
                except IndexError:
                    print("IndexError")
                    time.sleep(2)
                    continue
                except AttributeError:
                    print("AttributeError")
                    time.sleep(2)
                    continue
                except TypeError:
                    print("TypeError")
                    time.sleep(2)
                    continue
                except requests.exceptions.ConnectionError:
                    time.sleep(10)
                    continue
                time.sleep(2)
    time.sleep(60)
    page = page + 1
