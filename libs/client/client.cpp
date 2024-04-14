#include <iostream>
#include <boost/asio.hpp>
#include "client.hpp"

using boost::asio::ip::tcp;

Client::Client(boost::asio::io_context& io_context, std::string ip, unsigned short port)
    : _io_context(io_context), _endpoint(boost::asio::ip::address::from_string(ip), port) {

    //Class tcp::socket has no default constructor so we can't declare tcp::socket _socket;
    //Instead we resort to making a unique pointer.
    _socket = std::make_unique<tcp::socket>(_io_context);
}

void Client::connect() {
    _socket->connect(_endpoint);
}

void Client::write(unsigned request) {
    boost::asio::write(*_socket, boost::asio::buffer(&request, sizeof(request)));
}

void Client::read(std::vector<std::uint8_t>& values) {
    boost::asio::read(*_socket, boost::asio::buffer(values, 640 * 480 * 3));
}
