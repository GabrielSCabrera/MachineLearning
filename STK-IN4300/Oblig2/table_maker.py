def A():
    label_descr = { "ALTER": "Age", "ADHEU": "Allergic Coryza", "SEX": "Gender",
    "HOCHOZON": "High Ozone Village", "AMATOP": "Maternal Atopy",
    "AVATOP": "Paternal Atopy", "ADEKZ": "Neurodermatitis", "ARAUCH": "Smoker",
    "AGEBGEW": "Birth Weight", "FSNIGHT": "Night/Morning Cough",
    "FLGROSS": "Height", "FMILB": "Dust Sensitivity", "FNOH24": "Max. NO2",
    "FTIER": "Fur Sensitivity", "FPOLL": "Pollen Sensitivity",
    "FLTOTMED": "No. of Medis/Lufu", "FO3H24": "24h Max Ozone Value",
    "FSPT": "Allergic Reaction", "FTEH24": "24h Max Temperature",
    "FSATEM": "Shortness of Breath", "FSAUGE": "Itchy Eyes", "FLGEW": "Weight",
    "FSPFEI": "Wheezy Breath", "FSHLAUF": "Cough"}

    string = "\\begin{table}[H]\n\\center\n\t\\begin{tabular}{l c}\n\t"
    string += "\\textbf{Feature} & \\textbf{Description} \\\\ \n\t"
    string += "\\hline\n"
    for n,(k,v) in enumerate(label_descr.items()):
        string += f"\t{k} & {v}"
        if n < len(label_descr) - 1:
            string += "\\\\"
        string += "\n"
    string += "\t\\end{tabular}\n\\end{table}"

    print(string)

def B():
    label_descr = {"pregnant":"Number of Pregnancies",
    "glucose":"Plasma Glucose Concentration",
    "pressure":"Diastolic Blood Pressure", "triceps":"Triceps Skin Fold Thickness",
    "insulin":"2-H Serum Insulin", "mass":"Body Mass Index",
    "pedigree":"Diabetes Pedigree Function", "age":"Age"}

    string = "\\begin{table}[H]\n\\center\n\t\\begin{tabular}{l c}\n\t"
    string += "\\textbf{Feature} & \\textbf{Description} \\\\ \n\t"
    string += "\\hline\n"
    for n,(k,v) in enumerate(label_descr.items()):
        string += f"\t{k} & {v}"
        if n < len(label_descr) - 1:
            string += "\\\\"
        string += "\n"
    string += "\t\\end{tabular}\n\\end{table}"

    print(string)

def C():
    a = ["FLGROSS", "SEX", "FLGEW", "FO3H24", "FTEH24", "FPOLL", "FLTOTMED",
    "FSNIGHT", "FTIER", "FNOH24", "FSPFEI", "FSHLAUF", "AGEBGEW", "ARAUCH", "FSPT",
    "ADEKZ", "HOCHOZON", "FMILB", "FSAUGE", "ALTER", "AMATOP", "ADHEU", "FSATEM",
    "AVATOP"]

    b = [1.83E-01, -1.04E-01, 7.47E-02, 5.85E-02, -5.23E-02, -4.53E-02,
    -1.53E-02, 1.89E-02, -1.97E-02, -2.26E-02, 2.32E-02 , -1.32E-02, 1.46E-02,
    1.50E-02, 3.30E-02, 1.31E-02, -1.85E-02, -1.54E-02, -1.14E-02, 9.84E-03,
    5.69E-03, -5.44E-03, 3.70E-03, -3.57E-04]

    c = [1.06E-01, 6.63E-02, 9.99E-02, 1.33E-01, 1.24E-01, 1.33E-01, 5.31E-02,
    6.73E-02, 7.53E-02, 9.55E-02, 9.88E-02, 6.13E-02, 6.84E-02, 7.08E-02,
    1.58E-01, 6.52E-02, 9.46E-02, 8.92E-02, 7.33E-02, 8.08E-02, 7.20E-02,
    6.96E-02, 9.36E-02, 7.04E-02]

    d = [8.53E-02, 1.18E-01, 4.55E-01, 6.62E-01, 6.73E-01, 7.34E-01, 7.74E-01,
    7.79E-01, 7.94E-01, 8.13E-01, 8.14E-01, 8.30E-01, 8.31E-01, 8.33E-01, 8.35E-01,
    8.41E-01, 8.45E-01, 8.63E-01, 8.77E-01, 9.03E-01, 9.37E-01, 9.38E-01, 9.69E-01,
    9.96E-01]

    e = ["Height", "Gender", "Weight", "24h Max Ozone Value",
    "24h Max Temperature", "Pollen Sensitivity", "No. of Medis/Lufu",
    "Night/Morning Cough", "Fur Sensitivity", "Max. NO2", "Wheezy Breath",
    "Cough", "Birth Weight", "Smoker", "Allergic Reaction", "Neurodermatitis",
    "High Ozone Village", "Dust Sensitivity", "Itchy Eyes", "Age",
    "Maternal Atopy", "Allergic Coryza", "Shortness of Breath", "Paternal Atopy"]

    string = "\\begin{table}[H]\n\\center\n\t\\begin{tabular}{l l l l}\n\t"
    string += "\\textbf{Feature} & \\textbf{Coefficient}& \\textbf{Standard Error} & \\textbf{P-Value} \\\\ \n\t"
    string += "\\hline\n"
    for n,(i,j,k,l,m) in enumerate(zip(a,b,c,d,e)):
        string += f"\t{i} & {j:.2f} & {k:.2f} & {l:.2f}"
        if n < len(a) - 1:
            string += "\\\\"
        string += "\n"
    string += "\t\\end{tabular}\n\\end{table}"
    print(string)

def D():
    a = ["FLGROSS", "SEX", "FLGEW", "FO3H24", "FTEH24", "FPOLL", "FLTOTMED",
    "FSNIGHT", "FTIER", "FNOH24", "FSPFEI", "FSHLAUF", "AGEBGEW", "ARAUCH", "FSPT",
    "ADEKZ", "HOCHOZON", "FMILB", "FSAUGE", "ALTER", "AMATOP", "ADHEU", "FSATEM",
    "AVATOP"]

    b = [1.83E-01, -1.04E-01, 7.47E-02, 5.85E-02, -5.23E-02, -4.53E-02,
    -1.53E-02, 1.89E-02, -1.97E-02, -2.26E-02, 2.32E-02 , -1.32E-02, 1.46E-02,
    1.50E-02, 3.30E-02, 1.31E-02, -1.85E-02, -1.54E-02, -1.14E-02, 9.84E-03,
    5.69E-03, -5.44E-03, 3.70E-03, -3.57E-04]

    c = [1.06E-01, 6.63E-02, 9.99E-02, 1.33E-01, 1.24E-01, 1.33E-01, 5.31E-02,
    6.73E-02, 7.53E-02, 9.55E-02, 9.88E-02, 6.13E-02, 6.84E-02, 7.08E-02,
    1.58E-01, 6.52E-02, 9.46E-02, 8.92E-02, 7.33E-02, 8.08E-02, 7.20E-02,
    6.96E-02, 9.36E-02, 7.04E-02]

    d = [8.53E-02, 1.18E-01, 4.55E-01, 6.62E-01, 6.73E-01, 7.34E-01, 7.74E-01,
    7.79E-01, 7.94E-01, 8.13E-01, 8.14E-01, 8.30E-01, 8.31E-01, 8.33E-01, 8.35E-01,
    8.41E-01, 8.45E-01, 8.63E-01, 8.77E-01, 9.03E-01, 9.37E-01, 9.38E-01, 9.69E-01,
    9.96E-01]

    e = ["Height", "Gender", "Weight", "24h Max Ozone Value",
    "24h Max Temperature", "Pollen Sensitivity", "No. of Medis/Lufu",
    "Night/Morning Cough", "Fur Sensitivity", "Max. NO2", "Wheezy Breath",
    "Cough", "Birth Weight", "Smoker", "Allergic Reaction", "Neurodermatitis",
    "High Ozone Village", "Dust Sensitivity", "Itchy Eyes", "Age",
    "Maternal Atopy", "Allergic Coryza", "Shortness of Breath", "Paternal Atopy"]

    string = "\\begin{table}[H]\n\\center\n\t\\begin{tabular}{l l l l}\n\t"
    string += "\\textbf{Feature} & \\textbf{Coefficient}& \\textbf{Standard Error} & \\textbf{P-Value} \\\\ \n\t"
    string += "\\hline\n"
    for n,(i,j,k,l,m) in enumerate(zip(a,b,c,d,e)):
        string += f"\t{i} & {j:.2f} & {k:.2f} & {l:.2f}"
        if n < len(a) - 1:
            string += "\\\\"
        string += "\n"
    string += "\t\\end{tabular}\n\\end{table}"
    print(string)
