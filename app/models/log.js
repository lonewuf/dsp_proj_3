const mongoose = require('mongoose');

const logSchema = mongoose.Schema({
  name: String,
  violation: String,
  scl_pts: Number,
  date: {
  	type: Date,
  	// default: Date.now()
  }
});

module.exports = mongoose.model('Log', logSchema);