const express       = require('express'),
      bodyParser    = require('body-parser'),
      path          = require('path'),
      mongoose      = require('mongoose');

// Initialize app 
const app = express();

// Initialize dabatabase
mongoose.connect('mongodb://localhost/proj1db')
  .then(() => console.log('Database is running'))
  .catch(err => console.log(err));

// Import Models
const Student = require('./models/student');
const Log = require('./models/log');

app.use(express.static(path.join(__dirname, 'public')))
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({extended: true}));

app.get('/', (req, res) => {
  
  Student.find({}, function(err, foundStudents) {
    if(err)
    {
      throw (err);
    }
    else
    {
      res.render('students', {students: foundStudents});
    }
  });
});

app.get('/violation-log', (req, res) => {
  
  Log.find({})
    .then(response1 => {
      res.render('log', {students: response1});
    })
    .catch(err => console.log(err));
});



app.get('/fill-db', (req, res) => {

  var myarr = ['ABBY KATE C. ORIOQUE', 'ABIGAIL  ROSE  U. BARRA', 'ALLEN AUSTIN P. LANZADERAS', 
  'Ami Morita', 'Carlson Tabije', 'Charlene Mae Esteban', 'Ellene Gay S. Daniel', 'Erickson Calvo', 
  'Gia Marielle Reyes', 'Gil Panican Jr', 'John Nico Austria', 'Jose Alfonso Marquez', 'Kyle Adrian Lainez', 
  'Ma. Rowa Jean P. Tomonong', 'Marianne Bernardino', 'Paul Richard Villarete', 
  'RENATO JR. K. BALDEO', 'Sophia Vasquez'];

  for(var i = 0; myarr.length > i; i++)
  {
    var studArr = {
      name: myarr[i],
      std_num: i,
      scl_pts: 100,
      violation: ''
    }

    Student.create(studArr)
      .then(data => console.log("Success"))
      .catch(err => console.log(err));

  }

  res.send('sucess')

});

const port = 3000;

app.listen(port, (req, res) =>
  console.log(`Server is running on port ${port}`)
);