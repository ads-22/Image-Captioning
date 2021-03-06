from flask import Flask,render_template,redirect,request
import captionit

# __name__ == __main__
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def caption():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)# ./static/images.jpg        f.save(path)
        f.save(path)
        caption = captionit.caption_this_image(path)

        result_dic ={
            'image' : path,
            'caption' : caption
        }

        print(caption)
    
    return render_template("index.html",your_result=result_dic)

if __name__=='__main__':
    app.run(debug=True)