const mongoose = require('mongoose');

const tempSchema = mongoose.Schema({
  name: String
});

module.exports = mongoose.model('Temp', tempSchema);