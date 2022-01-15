#losses
pd.DataFrame(model.history.history).plot()

model.save('h5/'+fileName+'.h5')

testPR=model.predict(Xtest)
print(testPR)

#儲存
with open('TEST110002016_ver18.csv','w') as f:
  f.write('id,price\n')
  for i in range(len(testPR)):
    f.write(str(i+1)+","+str(float(testPR[i]))+"\n")
