# Lord of Heroes - Equipment Video Parser

## Install

  - Install [tesseract](https://tesseract-ocr.github.io/)
  - `pip install -r requirements.txt`

## Use

  - `python main.py [video_path]`
  - Video example: [link](https://drive.google.com/drive/folders/1BZuBRNbe4Qrea1koITLNrVnIqiNAbD8O?usp=sharing)
  - Output
```
[1396  198  521  533]
56
('체력', '47%') [('공격력', '5%'), ('체력', '675'), ('속도', '12'), ('방어력', '37')]
('방어력', '11%') [('치명타피해', '5%'), ('속도', '5'), ('효과적중', '4%'), ('체력', '5%')]
('치명타확률', '7%') [('속도', '4'), ('공격력', '5%'), ('공격력', '16'), ('방어력', '10')]
('효과적중', '12%') [('효과저항', '7%'), ('치명타피해', '4%'), ('체력', '6%'), ('방어력', '19')]
...
```
