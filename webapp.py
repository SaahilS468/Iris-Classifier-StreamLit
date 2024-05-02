import streamlit as st
import pickle as pk

lin = pk.load(open('lin.pkl','rb'))
log = pk.load(open('log.pkl','rb'))
svc = pk.load(open('svc.pkl','rb'))

def classify(num):
    if num < 0.5:
        return "Setosa"
    elif num < 1.5:
        return "Versicolor"
    else:
        return "Viginica"

def main():
    st.title("Iris Flower Clasification")

    activities = ["Linear Regression", "Logistic Regression", "SVM"]
    option = st.sidebar.selectbox("Which model would you like to use", activities)
    st.subheader(option)
    sl = st.slider("Select Sepal Length", 0.0, 10.0)
    sw = st.slider("Select Sepal Width", 0.0, 10.0)
    pl = st.slider("Select Petal Length", 0.0, 10.0)
    pw = st.slider("Select Petal Width", 0.0, 10.0)

    inputs = [[sl, sw, pl, pw]]

    if st.button("Classify"):
        if option == "Linear Regression":
            st.success(classify(lin.predict(inputs)))
        elif option == "Logistic Regression":
            st.success(classify(log.predict(inputs)))
        else:
            st.success(classify(svc.predict(inputs)))

if __name__ == "__main__":
    main()