set -x
rm -rf dist
NODE_ENV=production webpack

cd dist
git init
git remote add origin git@github.com:gregtatum/ml-py.git
git checkout -b gh-pages
git add .
git commit -m 'Deploy'
git push origin gh-pages -f
