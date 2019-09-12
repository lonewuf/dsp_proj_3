const mongoose = require('mongoose');

const studentSchema = mongoose.Schema({
  name: String,
  std_num: String,
  violation: String,
  scl_pts: Number
});

module.exports = mongoose.model('Student', studentSchema);