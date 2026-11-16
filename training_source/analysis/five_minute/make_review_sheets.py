"""Contact sheets of EXISTING train-only peak plots for all 18 flagged stations."""
from PIL import Image, ImageDraw
import pandas as pd

from pipeline import OUT


def main():
    rows=pd.read_csv(OUT/'station_selection.csv')
    ids=list(rows.loc[rows.category.eq('review_quality'),'deployment_id'])
    folder=OUT/'review_18'/'contact_sheets'
    folder.mkdir(parents=True,exist_ok=True)
    for offset in range(0,len(ids),6):
        sheet=Image.new('RGB',(1800,1080),'white')
        draw=ImageDraw.Draw(sheet)
        for j,station in enumerate(ids[offset:offset+6]):
            left=(j%2)*900
            top=(j//2)*360
            draw.text((left+10,top+5),station,fill='black')
            image=Image.open(OUT/'plots'/f'{station}_train_review.png').convert('RGB')
            image.thumbnail((890,330))
            sheet.paste(image,(left+5,top+25))
        path=folder/f'train_peak_sheet_{offset//6+1}.png'
        sheet.save(path)
        print(path)


if __name__=='__main__':
    main()
