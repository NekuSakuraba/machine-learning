# Dependencies

* scipy(>=0.13.3)
* numpy(>=1.8.2)
* scikit-learn(>=0.19.0)

# imbalanced-learn (0.3.0.dev0)
git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
cd imbalanced-learn
pip install .

# plotly (2.0.11)
conda install plotly

In order to use plotly, you should:
  * create an account "https://plot.ly/";
  * generate an API key on account settings; (settings -> API keys)
  * update the file "\home\.plotly\.credentials" with your API key;

The content should be like...
```
{
    "username": "USERNAME", 
    "stream_ids": [], 
    "api_key": "MY_API_KEY", 
    "proxy_username": "", 
    "proxy_password": ""
}
```