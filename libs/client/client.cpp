#include <iostream>
#include <boost/asio.hpp>
#include "client.hpp"

using boost::asio::ip::tcp;

/** @brief Writes data to the server.
*
* @param io_context The IO context object
* @param ip The IP address to connect to
* @param port The port number
* @return None
*/
Client::Client(boost::asio::io_context& io_context, std::string ip, unsigned short port)
    : _io_context(io_context), _endpoint(boost::asio::ip::address::from_string(ip), port) {

    //Class tcp::socket has no default constructor so we can't declare tcp::socket _socket;
    //Instead we resort to making a unique pointer.
    _socket = std::make_unique<tcp::socket>(_io_context);
}

/** @brief Connects the client to the server.
*
* @return None
*/
void Client::connect() {
    _socket->connect(_endpoint);
}

/** @brief Writes data to the server.
*
* @param request The request data to the server.
* @return None
*
* NOTE: This code will be changed.
*
*/
void Client::write(unsigned request) {
    boost::asio::write(*_socket, boost::asio::buffer(&request, sizeof(request)));
}

/** @brief Reads an image frame back from the server
*
* @param values The values of the image frame.
* @return None
*/
void Client::read(std::vector<std::uint8_t>& values) {
    boost::asio::read(*_socket, boost::asio::buffer(values, 640 * 480 * 3));
}
