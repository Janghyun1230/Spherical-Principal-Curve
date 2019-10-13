코드의 경우 python으로 짜여져 있으며, numpy (데이터 처리), torch (gradient descent)
, pandas (데이터 읽기), openpyxl (결과 저장) 라이브러리에 기반하고 있습니다.

Final Code.IPYNB는 interactive하게 코드 결과를 확인할 수 있습니다.

이 코드를 python file로 따로 정리하였습니다. 
- utils.py: 여러 함수들 모듈입니다. 
- principal_curves.py: principal curve fitting 및 plotting에 관련된 함수들 모듈입니다.
- main.py: For Earthquake data (fitted curve is saved)
python main.py -q 0.3 -e True -i False 
''' -q : neighborhood ratio
   -e : exact projection (ours) or not (Hauberg's)
   -i  : intrinsic expectation or not
'''

- silmulation.py: simulation data experiment reproduce (excel files are saved, circle과 wave):
python simulation.py

