import pandas as pd
import numpy as np
import datetime as dt
from csv import writer
from sklearn.impute import KNNImputer
import pickle


def clean_pred(dirty_df):
    # get tlv rows
    dirty_df = dirty_df[dirty_df['linqmap_city'] == 'תל אביב - יפו'].copy()

    # remove unwanted cols
    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments', 'linqmap_city']
    dirty_df.drop(columns=cols_to_remove, inplace=True)

    # fill nans
    dirty_df.fillna(method='ffill', inplace=True)
    dirty_df.fillna(method='bfill', inplace=True)

    # add NS & WE cols
    dirty_df['NS'] = np.where((dirty_df['linqmap_magvar'] < 90) | (dirty_df['linqmap_magvar'] > 270), 1, 0)
    dirty_df['EW'] = np.where((dirty_df['linqmap_magvar'] < 180), 1, 0)

    # convert date & time & timestamp to datetime object
    dirty_df['pubDate'] = dirty_df.apply(lambda row: dt.datetime.strptime(row.pubDate, "%m/%d/%Y %H:%M:%S"), axis=1)
    dirty_df['update_date'] = dirty_df.apply(lambda row: dt.datetime.fromtimestamp(row.update_date / 1000), axis=1)
    dirty_df['event_time_hours'] = dirty_df.apply(lambda row: (row.update_date - row.pubDate).total_seconds() / 3600,
                                                  axis=1)
    initial = dirty_df.shape[0]
    streets = ['המכבי', 'בן יהודה', 'לבונטין', 'לילינבלום', 'אייזיק רמבה', 'אליפלט', 'השומרון', 'רציף הרברט סמואל',
               'התערוכה',
               'דרך שלמה', 'העליה', 'יהודית', 'סלומון', 'יוסף לוי', 'הארבעה', 'הברזל', 'יצחק שדה', 'ארלוזורוב',
               'הגלבוע',
               'קפריסין', 'עליית הנוער', 'הגבורה', 'אד קוץ', 'חומה ומגדל', 'בני אפרים', 'פרופסור יחזקאל קויפמן',
               'יהודה מרגוזה', 'ריב"ל', 'החרש', 'מלכי ישראל', 'יצחק טבנקין', 'ח"ן', 'בוגרשוב', 'עודד', 'בלוך',
               'לוי אשכול',
               'ש"י עגנון', 'יצחק גרציאני', 'השוק', 'י.ל. גורדון', 'וילנא', 'שפ"ר', 'מזא"ה', 'שיקוע קרליבך',
               'שד קרן קיימת לישראל', 'היינריך היינה', 'צבי פרייגרזון', 'רוני', 'אבן גבירול', 'ירוחם משל', 'קלישר',
               'אלנבי',
               'רחל', 'הדר יוסף', 'יחזקאל', 'אהרון בקר', 'לאונרדו דה וינצי', 'דרך בגין', 'יפת', 'תובל', 'גבעון',
               'ברנדייס',
               'שבתאי דון יחיא', 'זלוציסטי', 'רמז', 'חיים לבנון', 'יוניצמן', 'אחד העם', 'פינס', 'הטייסים',
               'ארניה אוסוולדו',
               'הפרדס', 'מח"ל', 'שגב', 'שמחה בלאס', 'ישראל מסלנט', 'אליעזר פרי', 'רש"י', 'הנריטה סאלד', 'שארית ישראל',
               'לוחמי גליפולי', 'הכרמל', 'ראול ולנברג', 'שלבים', 'דוד רזיאל', 'יבנה', 'הרב ניסים', 'דפנה', 'בצלאל יפה',
               'לואי פסטר', 'שטרית (מנהרה)', 'יהודה הימית', 'מונטיפיורי', 'גבעת התחמושת', 'מנהרת מנחם בגין',
               'טרומפלדור',
               'בן סרוק', 'לח"י', 'ירמיהו', 'ויצמן', 'לה גווארדיה', 'לויד גורג', 'דבורה הנביאה', 'בצרון', 'ריינס',
               'דיזנגוף',
               'שינקין', 'שד רוקח', 'דרך יפו', 'מוהליבר', 'שד רוקח (גשר)', 'יד חרוצים', 'שד ירושלים', 'רא"ל יעקב דורי',
               'בית אשל', 'ציונה תגר', 'המלך גורג', '3780', 'העבודה', 'תל גיבורים', 'פנחס רוזן', 'שטראוס',
               'נחלת בנימין',
               'גרוזנברג', 'משמר הירדן', 'נחמה', 'יהושוע בן נון', 'יוחנן הסנדלר', 'שמריהו לוין', 'דרך חיל השריון',
               'זבוטינסקי', 'גשר יצחק מודעי', 'מוזס יהודה ונח', 'סירקין', 'פוריה', 'ברגסון', 'אילת', 'הגיבור האלמוני',
               'אוסישקין', 'סעדיה גאון', 'על פרשת דרכים', 'מאפו', 'נחלת יצחק', 'לבנדה', 'שטרית', 'יהודה הלוי',
               'משה סנה',
               'האומנים', 'מעפילי אגוז', 'יגאל אלון', 'רבנו חננאל', 'שניאור', 'אליעזר קפלן', 'ינון', 'הרב שלמה גורן',
               'נורדאו', 'פנחס ספיר', 'אברהם שלונסקי', 'הרצל', 'דרך השלום', 'הר ציון', 'לסקוב חיים', 'משה דיין',
               'עולי ציון',
               'קרית שאול', 'פנקס', 'פרישמן', 'וייס אייזיק הירש', 'בן צבי', 'דרך אריה (לובה) אליאב', 'מאנגר איציק',
               'החשמונאים', 'הוברמן', 'לוינסקי', 'שתולים', 'רקנאטי', 'טשרניחובסקי', 'הרכבת', 'קרליבך', 'צה"ל',
               'עמק ברכה', 'צבי פרופס', 'בני דן', 'בן גוריון', 'דרך נמיר', 'איסרליש', 'ה באייר', 'קוסובסקי', 'הורודצקי',
               'המלך כורש', '2040', 'רוטשילד', 'הכובשים', 'צלנוב', 'יד לבנים', '976', 'פנחס לבון', 'הרב הלר חיים',
               'המרד',
               'דם המכבים', 'ההגנה', 'שונצינו', 'ניצנה', 'ערבי נחל', 'רנ"ק', 'השחר', 'מרזוק ועזר', 'קהילת יאסי',
               'וינגייט',
               'מויאל אהרון', 'פינסקר', '1185', 'גואל', 'קלאוזנר', 'דרך חיים בר-לב', 'הירקון', 'בנימיני',
               'דרך קיבוץ גלויות', 'המסגר',
               'איינשטיין', 'ידידיה פרנקל', 'ברזילי', 'ביל"ו', 'נחום גולדמן', 'גלוסקא', 'המלאכה', 'שד שאול המלך',
               'קלצקין', 'לינקולן', 'הצפירה', 'יגיע כפיים', 'הרכב']
    dirty_df = dirty_df[['linqmap_street'] + [col for col in dirty_df.columns if col != 'linqmap_street']]
    for i in range(len(streets)):
        lst = [None] * (dirty_df.shape[1])
        lst[0] = streets[i]
        dirty_df.loc[dirty_df.shape[0]] = lst

    # add dummies for street
    dums = pd.get_dummies(dirty_df['linqmap_street'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)

    dirty_df = dirty_df.iloc[:initial,:].copy()
    print(dirty_df)

    # create saved df file of types and subtype
    subtypes = dirty_df.groupby('linqmap_type').aggregate({'linqmap_subtype': np.unique})
    subtypes.to_csv('subtypes.csv')

    dirty_df.sort_values(by='update_date', inplace=True)

    # remove magvar & pubdate & street colb
    dirty_df.drop(columns=['linqmap_magvar', 'pubDate', 'linqmap_street', 'update_date', 'linqmap_type'], inplace=True)
    # add dummies for subtype
    dums = pd.get_dummies(dirty_df['linqmap_subtype'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)
    return dirty_df


def get_comb_for_pred(df):
    new_cols = []
    for i in range(1, 5):
        new_cols += list(df.columns + '_S' + str(i))
    with open('to_predict.csv', 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_cols)
        for i in range(0, df.shape[0], 4):
            print('\r', str(round(i / (df.shape[0] - 4) * 100, 2)) + ' %', end='')
            cur_row = []
            cur_row += list(df.iloc[i, :])
            cur_row += list(df.iloc[i + 1, :])
            cur_row += list(df.iloc[i + 2, :])
            cur_row += list(df.iloc[i + 3, :])
            writer_object.writerow(cur_row)
        print('\r100 %')
    f_object.close()
    temp = pd.read_csv('to_predict.csv')
    temp = temp.reindex(sorted(temp.columns), axis=1)
    print(temp.shape)
    return temp


def preprocess(dirty_df):
    dirty_df = clean_pred(dirty_df)
    return get_comb_for_pred(dirty_df)


def run_1(data_path):
    df = preprocess(pd.read_csv(data_path))

    # load the model from disk
    loaded_model = pickle.load(open('finalized_knn_model.sav', 'rb'))

    # read trained model
    # output results
    return
