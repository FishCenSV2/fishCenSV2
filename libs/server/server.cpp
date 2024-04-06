#include <iostream>
#include "server.hpp"
#include "session.hpp"

using boost::asio::ip::tcp;

Server::Server(boost::asio::io_context& io_context, unsigned short port, std::mutex& mutex)
    : _io_context(io_context), _acceptor(io_context, tcp::endpoint(tcp::v4(), port)), mutex(mutex) {
    do_accept();

}

void Server::run() {
    _io_context.run();
}

void Server::stop() {
    _io_context.stop();
}

/** @brief Accepts incoming connection from client
*
* @return None
*/
void Server::do_accept() {
    _acceptor.async_accept([this](boost::system::error_code ec, tcp::socket socket) {
        if (!ec) {
            std::cout << "Connection accepted. Creating session\n";

            std::make_shared<Session>(std::move(socket), *this)->start();
        }

        do_accept();
        });
}