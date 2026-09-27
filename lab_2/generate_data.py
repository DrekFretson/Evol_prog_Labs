import csv, random
from pathlib import Path
SEED=20260927
NAMES=['Осциллограф', 'Лабораторный БП', 'Мультиметр', 'Генератор сигналов', 'Логический анализатор', 'Микроскоп', 'Спектрометр', 'Термокамера', 'Паяльная станция', 'Вытяжка', '3D-принтер', 'ЧПУ-станок', 'Компрессор', 'Вакуумный насос', 'Весы', 'ПК рабочей станции', 'Сервер вычислений', 'ИБП', 'Сетевой коммутатор', 'Монитор', 'Набор датчиков', 'DAQ-модуль', 'Калибратор', 'Источник опорного напряжения', 'LCR-метр', 'Шкаф безопасности', 'Лазерный модуль', 'Оптический стол', 'Камера высокого разрешения', 'Робот-манипулятор', 'Система вентиляции', 'Холодильная камера', 'Ультразвуковая ванна', 'Центрифуга', 'Стабилизатор напряжения', 'Прецизионный термометр', 'Датчик давления', 'Система освещения', 'Электронная нагрузка', 'Частотомер']
CATEGORIES=['Измерение', 'Вычисления', 'Оптика', 'Электроника', 'Механика', 'Безопасность']
CONFLICTS=[(6, 27), (7, 27), (11, 12), (13, 14), (16, 17), (17, 30), (18, 35), (28, 30), (31, 32), (33, 34), (9, 26), (21, 29)]
def main():
    rng=random.Random(SEED)
    root=Path(__file__).resolve().parent
    d=root/"data"; d.mkdir(exist_ok=True)
    rows=[]
    for i,name in enumerate(NAMES,1):
        rows.append([i,name,rng.randrange(25,251)*1000,rng.randrange(1,31)*100,
                     rng.randrange(20,101),CATEGORIES[(i-1)%len(CATEGORIES)]])
    with open(d/"equipment.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["id","name","price_rub","power_w","utility","category"]); w.writerows(rows)
    with open(d/"conflicts.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["item_a","item_b"]); w.writerows(CONFLICTS)
    print("Данные сгенерированы. seed =",SEED)
if __name__=="__main__": main()
