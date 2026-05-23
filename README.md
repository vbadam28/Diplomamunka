# Diplomamunka

## Környezet

### Python verzió
- Python 3.10.2

### Felhasznált külső csomagok
- PySide6: 6.9.0
- numpy: 2.2.6
- opencv: 4.12.0
- matplotlib: 3.10.6
- nibabel: 5.3.2
- scipy: 1.15.3
- seaborn: 0.3.12 (eredmények értelmezésénél)
- pandas: 2.3.3 (eredmények értelmezésénél)
- tqdm: 4.66.2 (kiértékelésnél)

---

## Projekt futtatása

A fő alkalmazás indítása:

```bash
python main.py
```

### Tesztadatok
A `data_ni` mappában található néhány tesztkép a program kipróbálásához.

---

## Eredmények/Statisztikák vizualizálása (6. fejezetben ábrák)

Az elkészült eredmények megtekintése:

```bash
python results.py
```

### Megjegyzés
- A szkriptet az `evaluate` mappából érdemes futattni
- A szkript az `evaluate/csv` mappából olvassa be a CSV fájlokat.
- Fájlon belül lehet állítani a csv fájlok elérési útvonalát.
- A csv fájlokban szereplő kép indexeknél az i. kép, az egyel kevesebb, mint az eredeti adathalmazban a megfelelője.
    - Tehát csv-ben {i}. kép az valójában BraTS2020\_Training\_ {i+1} \_flair.nii
    - Hasonlóan a csv-ben a rétegek szintén 0-tól vannak indexelve, de az alkalmazásban 1-től számolom a rétegeket 
---

## CSV fájlok újraszámolása

A kiértékelő CSV fájlok újraszámolása:

```bash
python compute_result.py
```

### Megjegyzések
- A szkriptet az `evaluate` mappából érdemes futattni
- Az új CSV fájlok az `evaluate/csv` mappába kerülnek.
- Fájlon belül lehet beállítani, hogy milyen pipeline-okat futasson és milyen néve mentsen
- Teljes adathalmazon fut
- A kiértékeléshez használt teljes adathalmaz, megfelelő formátumban elérhető:  
  `[https://drive.google.com/drive/folders/1EZZz1IwF9RJtUu4z89xD8BfqqUT15NO_]`
- ( formátum:  mappa: brain_{i} , fájlok: {i}_flair.nii, {i}_seg.nii  )
- ({i} kép indexe.  pl.: {i}\_flair.nii eredetije --> BraTS2020\_Training\_{i+1}\_flair.nii)
---
