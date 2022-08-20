const changeBackgroundRGB = ({ element, red, gree, blue }) => {
  element.style.setProperty('--red', red);
  element.style.setProperty('--green', gree);
  element.style.setProperty('--blue', blue);
 };

 export default changeBackgroundRGB;